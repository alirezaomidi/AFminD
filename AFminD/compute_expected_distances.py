import click
import numpy as np
import scipy.special
import concurrent.futures
from glob import glob
import os
import zipfile
import json
import pickle
from tqdm import tqdm
import logging
from .alphafold.common.confidence import _calculate_bin_centers

# set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def compute_predicted_distance(logits: np.ndarray, breaks: np.ndarray) -> np.ndarray:
    probs = scipy.special.softmax(logits.astype(np.float64), axis=-1)
    bin_centers = _calculate_bin_centers(breaks)

    # Tuple of expected aligned distance error and max possible error.
    return np.sum(probs * bin_centers, axis=-1)


def distogram_json(distogram: np.ndarray) -> str:
    if distogram.ndim != 2 or distogram.shape[0] != distogram.shape[1]:
        raise ValueError(f"Distogram must be a square matrix, got {distogram.shape}")

    # Round the predicted aligned errors to 2 decimal place.
    rounded_distances = np.round(distogram.astype(np.float64), decimals=2)
    formatted_output = {
        "predicted_distances": rounded_distances.tolist(),
    }

    return json.dumps(formatted_output, indent=None, separators=(",", ":"))


def compute_predicted_distances_from_zip(zip_file: str):
    with zipfile.ZipFile(zip_file, "a") as zip_ref:
        pickle_filenames = [f for f in zip_ref.namelist() if f.endswith(".pickle")]
        for pickle_filename in pickle_filenames:
            output_file = pickle_filename.replace(".pickle", ".distogram.json")
            if output_file in zip_ref.namelist():
                continue
            data = pickle.load(zip_ref.open(pickle_filename))
            # compute distogram
            distogram = compute_predicted_distance(
                logits=data["distogram"]["logits"],
                breaks=data["distogram"]["bin_edges"],
            )
            # save distogram as json back into zip file
            distogram_json_str = distogram_json(distogram)
            zip_ref.writestr(output_file, distogram_json_str)


@click.command()
@click.option(
    "-i",
    "--input",
    type=click.Path(dir_okay=True, file_okay=True),
    help="Colabfold output directory or a single job zip file",
    required=True,
)
@click.option("--n-jobs", type=int, default=1, help="Number of parallel jobs to run")
@click.option("--debug", is_flag=True, help="Print debug information", envvar="DEBUG")
def main(input, n_jobs, debug):
    # logger level
    logger.setLevel(logging.DEBUG if debug else logging.INFO)

    zip_files = []
    if os.path.isfile(input) and input.endswith(".result.zip"):
        zip_files.append(input)
    elif os.path.isdir(input):
        zip_files = glob(os.path.join(input, "*.result.zip"))
    else:
        raise ValueError(f"Invalid input: {input}. Must be a zip file or a directory")

    with concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = {
            executor.submit(compute_predicted_distances_from_zip, zip_file): zip_file
            for zip_file in zip_files
        }

        for fut in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(futures),
            desc="Computing pDistogram",
            unit="zip(s)",
        ):
            try:
                fut.result()
            except Exception as e:
                print(f"Error processing {futures[fut]}: {e}")


if __name__ == "__main__":
    main()
