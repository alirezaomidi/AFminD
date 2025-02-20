import click
import logging
import pandas as pd
import numpy as np
from glob import glob
import concurrent.futures
from tqdm import tqdm
import re
import zipfile
import string
import json
import os
from AFminD.utils import find_msa_file, get_chain_lengths, get_jobname
from AFminD.extract_scores import format_array


# set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def find_distogram_files(zip_path):
    # get jobname
    jobname = get_jobname(zip_path)

    # find pdb files in the zip
    df = []
    with zipfile.ZipFile(zip_path) as z:
        distogram_files = [f for f in z.namelist() if f.endswith(".distogram.json")]
        print(distogram_files)
        for distogram_file in distogram_files:
            pred_jobname, rank, model_name, model, seed, recycle = re.match(
                r"(.*)_all_rank_(\d+)_(.*)_model_(\d+)_seed_(\d+)(?:\.r(\d+))?\.distogram\.json",
                distogram_file,
            ).groups()
            pred_jobname = os.path.basename(pred_jobname)
            assert pred_jobname == jobname, f'Jobname "{jobname}" does not match regex'
            rank = int(rank)
            model = int(model)
            seed = int(seed)
            if recycle is not None:
                recycle = int(recycle)
            df.append(
                (
                    jobname,
                    rank,
                    model_name,
                    model,
                    seed,
                    recycle,
                    zip_path,
                    distogram_file,
                )
            )

    return pd.DataFrame(
        df,
        columns=[
            "jobname",
            "rank",
            "model_name",
            "model",
            "seed",
            "recycle",
            "zip_path",
            "distogram_file",
        ],
    )


def find_msa_and_distogram_files(zip_path):
    msa_file = find_msa_file(zip_path)
    distogram_files = find_distogram_files(zip_path)
    return pd.merge(msa_file, distogram_files, on=["zip_path", "jobname"], how="inner")


def read_distogram(zip_path, score_filename, a3m_filename):
    # read the scores
    with zipfile.ZipFile(zip_path) as z:
        with z.open(score_filename) as f:
            scores = json.load(f)
        with z.open(a3m_filename) as f:
            a3mhead = f.readline().decode("utf-8")

    # read the first line of the MSA file to extract chain lengths
    chain_lengths = get_chain_lengths(a3mhead)
    chain_indices = np.cumsum(chain_lengths)[:-1]

    # distogram
    distogram = np.array(scores["predicted_distances"])
    assert len(distogram) == sum(
        chain_lengths
    ), f"Distogram length does not match: {len(distogram)} != {sum(chain_lengths)}"

    # split the pae matrix into blocks
    distograms = np.split(distogram, chain_indices)
    for i, distogram in enumerate(distograms):
        distograms[i] = np.split(distogram, chain_indices, axis=1)
    # extract statistics from the pae matrix
    distograms = [
        [
            {
                "min": np.min(distogram, axis=0).tolist(),
            }
            for distogram in distogram_row
        ]
        for distogram_row in distograms
    ]

    scores = {}
    for distogram_row, chain_id in zip(distograms, string.ascii_uppercase):
        for pae, chain_id2 in zip(distogram_row, string.ascii_uppercase):
            for key, value in pae.items():
                scores[f"distogram_{chain_id}_{chain_id2}_{key}"] = format_array(value)

    return scores


@click.command()
@click.option(
    "-i",
    "--input",
    type=click.Path(dir_okay=True, file_okay=True, exists=True),
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
def main(input, n_jobs, output, from_scratch, debug):
    # logger level
    logger.setLevel(logging.DEBUG if debug else logging.INFO)

    # list predictions
    zip_files = []
    if os.path.isfile(input) and input.endswith(".result.zip"):
        zip_files.append(input)
    elif os.path.isdir(input):
        zip_files = glob(os.path.join(input, "*.result.zip"))
    else:
        raise ValueError(f"Invalid input: {input}. Must be a zip file or a directory")

    # run the jobs
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs) as executor:
        # first, find the pdb files
        futures = {
            executor.submit(find_msa_and_distogram_files, zip_file)
            for zip_file in zip_files
        }
        df_pred = []
        for fut in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Finding files",
            unit="entries",
        ):
            try:
                result = fut.result()
                df_pred.append(result)
            except Exception as e:
                logger.exception(e)
                pass
        df_pred = pd.concat(df_pred, ignore_index=True)

        # check if the output file already exists
        df = None
        if os.path.exists(output) and not from_scratch:
            df = pd.read_csv(output)
            logger.info(f"Output file already exists, skipping {len(df)} entries")
            df_pred = df_pred[~df_pred.jobname.isin(df.jobname)]

        # finally, read the scores
        futures = {
            executor.submit(
                read_distogram,
                row.zip_path,
                row.distogram_file,
                row.msa_filename,
            ): (
                row.jobname,
                row.rank,
                row.model_name,
                row.model,
                row.seed,
                row.recycle,
            )
            for row in df_pred.itertuples()
        }

        df_af_scores = []
        for i, fut in enumerate(
            tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc="Reading distograms",
                unit="files",
            )
        ):
            jobname, rank, model_name, model, seed, recycle = futures[fut]
            try:
                result = fut.result()
                result["jobname"] = jobname
                result["rank"] = rank
                result["model_name"] = model_name
                result["model"] = model
                result["seed"] = seed
                result["recycle"] = recycle
                df_af_scores.append(result)
            except Exception as e:
                logger.exception(e)

        # save the dataset
        df_af_scores = pd.DataFrame(df_af_scores)
        if df is not None:
            df_af_scores = pd.concat([df, df_af_scores], ignore_index=True)
        df_af_scores.to_csv(output, index=False)


if __name__ == "__main__":
    main()
