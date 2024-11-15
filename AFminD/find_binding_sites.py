# %%
import click
import logging
import pandas as pd
import numpy as np
from scipy import signal
from AFminD.extract_scores import format_array

# %%
# set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# %%
def get_binding_sites(df, prominence=0.02, distance=30):
    # find distogram_X_Y_min columns
    columns = df.columns
    columns = columns[columns.str.startswith("distogram") & columns.str.endswith("min")]

    df_peaks = []
    for col in columns:
        chain1, chain2 = col.split("_")[1:3]
        if chain1 == chain2:
            continue
        if df[col].isna().all():
            continue
        # convert the data to numpy array
        x = df[col].apply(lambda x: list(map(float, x.split(","))))
        x = np.array(x.tolist())

        # get the best possible distance for each residue in chain2
        x_score = x.min(axis=0)
        x_score = 1.0 - (x_score - 2.0) / 20.0

        # find peaks
        peaks, _ = signal.find_peaks(x_score, prominence=prominence, distance=distance)
        mask = np.zeros_like(x_score, dtype=bool)
        mask[peaks] = True
        scores = x_score[mask]
        idx = np.argsort(scores)[::-1]
        peaks = peaks[idx]
        scores = scores[idx]
        df_peaks.append(
            dict(
                chain1=chain1,
                chain2=chain2,
                peaks=format_array(peaks, fmt="{:d}"),
                scores=format_array(scores),
            )
        )

    return pd.DataFrame(df_peaks)


# %%
@click.command()
@click.option(
    "-i",
    "--input",
    type=click.Path(dir_okay=False, file_okay=True),
    help="A CSV file containing minDs",
    required=True,
)
@click.option(
    "-o",
    "--output",
    type=click.Path(dir_okay=False, file_okay=True),
    help="Output csv file",
    required=True,
)
@click.option(
    "--prominence", type=float, default=0.02, help="Prominence for peak finding"
)
@click.option("--distance", type=int, default=30, help="Minimum distance between peaks")
@click.option("--debug", is_flag=True, help="Print debug information", envvar="DEBUG")
def main(input, output, prominence, distance, debug):
    # logger level
    logger.setLevel(logging.DEBUG if debug else logging.INFO)

    # read the distogram file
    df = pd.read_csv(input)

    # get the binding sites
    df_binding_sites = (
        df.groupby("jobname")
        .apply(get_binding_sites, prominence=prominence, distance=distance)
        .reset_index()
        .drop("level_1", axis=1)
    )

    # save the binding sites
    df_binding_sites.to_csv(output, index=False)


# %%
if __name__ == "__main__":
    main()
