import click
import logging
import pandas as pd
import string
import numpy as np
from glob import glob
import concurrent.futures
from tqdm import tqdm
import re
import zipfile
import json
import os
from AFminD.utils import find_msa_file, get_chain_lengths, get_jobname, format_array


# set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def find_score_files(zip_path):
    # get jobname
    jobname = get_jobname(zip_path)

    # find pdb files in the zip
    df = []
    with zipfile.ZipFile(zip_path) as z:
        scores_files = [f for f in z.namelist() if f.endswith(".json") and "score" in f]
        for score_filename in scores_files:
            pred_jobname, rank, model_name, model, seed = re.match(
                r"(.*)_scores_rank_(\d+)_(.*)_model_(\d+)_seed_(\d+).*",
                score_filename,
            ).groups()
            assert pred_jobname == jobname, f'Jobname "{jobname}" does not match regex'
            rank = int(rank)
            model = int(model)
            seed = int(seed)
            df.append(
                dict(
                    jobname=jobname,
                    rank=rank,
                    model_name=model_name,
                    model=model,
                    seed=seed,
                    zip_path=zip_path,
                    score_filename=score_filename,
                )
            )
    return pd.DataFrame(df)


def read_scores(zip_path, score_filename, a3m_filename):
    # read the scores
    with zipfile.ZipFile(zip_path) as z:
        with z.open(score_filename) as f:
            scores = json.load(f)
        with z.open(a3m_filename) as f:
            a3mhead = f.readline().decode("utf-8")

    # read the first line of the MSA file to extract chain lengths
    chain_lengths = get_chain_lengths(a3mhead)
    chain_indices = np.cumsum(chain_lengths)[:-1]

    # plddt
    plddt = np.array(scores["plddt"])
    assert len(plddt) == sum(
        chain_lengths
    ), f"PLDDT length does not match: {sum(chain_lengths)} != {len(plddt)}"
    plddts = np.split(plddt, chain_indices)

    # pae
    pae = np.array(scores["pae"])
    assert len(pae) == sum(
        chain_lengths
    ), f"PAE length does not match: {sum(chain_lengths)} != {len(pae)}"
    # split the pae matrix into blocks
    paes = np.split(pae, chain_indices)
    for i, pae in enumerate(paes):
        paes[i] = np.split(pae, chain_indices, axis=1)
    # extract statistics from the pae matrix
    paes = [
        [
            {
                "min": np.min(pae, axis=0).tolist(),
                "mean": np.mean(pae, axis=0).tolist(),
            }
            for pae in pae_row
        ]
        for pae_row in paes
    ]

    # ptm, iptm, combined
    ptm = scores["ptm"]
    iptm = scores["iptm"]
    combined = 0.2 * ptm + 0.8 * iptm

    scores = {}
    for plddt, chain_id in zip(plddts, string.ascii_uppercase):
        scores[f"plddt_{chain_id}"] = format_array(plddt)
    for pae_row, chain_id in zip(paes, string.ascii_uppercase):
        for pae, chain_id2 in zip(pae_row, string.ascii_uppercase):
            for key, value in pae.items():
                scores[f"pae_{chain_id}_{chain_id2}_{key}"] = format_array(value)
    scores["ptm"] = ptm
    scores["iptm"] = iptm
    scores["combined"] = f"{combined:.2f}"

    return scores


def find_msa_and_score_files(zip_path):
    msa_file = find_msa_file(zip_path)
    score_file = find_score_files(zip_path)

    return pd.merge(msa_file, score_file, on=["zip_path", "jobname"], how="inner")


@click.command()
@click.option(
    "-d",
    "--colabfold-dir",
    type=click.Path(dir_okay=True, file_okay=False, exists=True),
    required=True,
)
@click.option(
    "--n-jobs",
    default=1,
    type=int,
    help="Number of jobs to run in parallel",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(dir_okay=False, file_okay=True),
    help="Output csv file",
    required=True,
)
@click.option(
    "--from-scratch",
    is_flag=True,
    help="If set, will recompute the scores from scratch",
    default=False,
)
@click.option("--debug", is_flag=True, help="Print debug information", envvar="DEBUG")
def main(colabfold_dir, n_jobs, output, from_scratch, debug):
    # logger level
    logger.setLevel(logging.DEBUG if debug else logging.INFO)

    # list predictions
    zip_files = glob(f"{colabfold_dir}/*.result.zip")

    # run the jobs
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs) as executor:
        # first, find the pdb files
        futures = {
            executor.submit(find_msa_and_score_files, zip_file)
            for zip_file in zip_files
        }
        df_pred = []
        for fut in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Finding files",
            unit="files",
        ):
            try:
                result = fut.result()
                df_pred.append(result)
            except Exception as e:
                logger.exception(e)
        df_pred = pd.concat(df_pred, ignore_index=True)

        # check if the output file already exists
        df = None
        if os.path.exists(output) and not from_scratch:
            df = pd.read_csv(output)
            logger.info(f"Output file already exists, skipping {len(df)} entries")
            df_pred = df_pred[~df_pred.jobname.isin(df.jobname.unique())]

        # finally, read the scores
        futures = {
            executor.submit(
                read_scores, row.zip_path, row.score_filename, row.msa_filename
            ): (row.jobname, row.rank, row.model_name, row.model, row.seed)
            for row in df_pred.itertuples()
        }

        df_scores = []
        for fut in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Reading scores",
            unit="files",
        ):
            jobname, rank, model_name, model, seed = futures[fut]
            try:
                result = fut.result()
                result["jobname"] = jobname
                result["rank"] = rank
                result["model_name"] = model_name
                result["model"] = model
                result["seed"] = seed
                df_scores.append(result)
            except Exception as e:
                logger.exception(e)

        df_scores = pd.DataFrame(df_scores)
        if df is not None:
            df_scores = pd.concat([df, df_scores], ignore_index=True)

        # save the dataset
        df_scores.to_csv(output, index=False)


if __name__ == "__main__":
    main()
