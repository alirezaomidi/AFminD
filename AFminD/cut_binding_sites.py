# %%
import click
import logging
import pandas as pd
import os
import zipfile
from AFminD.utils import parse_jobname, get_jobname
from AFminD.data.utils import read_fasta, write_fasta

# %%
# set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# %%
def read_sequence_csv_file(zip_path):
    jobname = get_jobname(zip_path)

    sequence_csv_file = f"{jobname}/{jobname}.csv"
    with zipfile.ZipFile(zip_path) as z:
        with z.open(sequence_csv_file) as f:
            return pd.read_csv(f)


# %%
def cut_segment(segment, start, end):
    if isinstance(segment, tuple):
        assert len(segment) == 2, f"Segment {segment} has more than two elements"
        return (segment[0] + start, segment[0] + end - 1)
    elif isinstance(segment, str) and segment.startswith("X"):
        return f"X{end - start}"
    else:
        raise ValueError(f"Unknown segment {segment}")


# %%
def cut_jobname(jobname, start, end, seq=None):
    jobname = parse_jobname(jobname)
    assert len(jobname) == 1, f"Jobname {jobname} has more than one chain"
    name, segments = jobname.pop()
    if not segments:
        assert (
            seq is not None
        ), "Sequence is required to determine the segments when jobname has no segment info"
        segments = [(1, len(seq))]

    segment_lengths = []
    for segment in segments:
        if isinstance(segment, tuple):
            assert len(segment) == 2, f"Segment {segment} has more than two elements"
            segment_lengths.append(segment[1] - segment[0] + 1)
        elif isinstance(segment, str) and segment.startswith("X"):
            segment_lengths.append(int(segment[1:]))
        else:
            raise ValueError(f"Unknown segment {segment}")
    segment_len_cumsum = [
        sum(segment_lengths[: i + 1], 0) for i in range(0, len(segment_lengths))
    ]

    for i, seg_len_cumsum in enumerate(segment_len_cumsum):
        if start < seg_len_cumsum:
            start_seg_index = i
            start_seg_offset = start - (seg_len_cumsum - segment_lengths[i])
            break
    else:
        raise ValueError(f"Start {start} is out of range {segment_len_cumsum}")

    for i, seg_len_cumsum in enumerate(segment_len_cumsum):
        if end <= seg_len_cumsum:
            end_seg_index = i
            end_seg_offset = end - (seg_len_cumsum - segment_lengths[i])
            break
    else:
        raise ValueError(f"End {end} is out of range")

    if start_seg_index == end_seg_index:
        new_segments = [
            cut_segment(segments[start_seg_index], start_seg_offset, end_seg_offset)
        ]
    else:
        new_segments = [
            cut_segment(
                segments[start_seg_index],
                start_seg_offset,
                segment_lengths[start_seg_index],
            ),
            *segments[start_seg_index + 1 : end_seg_index],
            cut_segment(segments[end_seg_index], 0, end_seg_offset),
        ]

    new_jobname = (
        name
        + "-"
        + "-".join(
            [
                f"{seg[0]}-{seg[1]}" if isinstance(seg, tuple) else seg
                for seg in new_segments
            ]
        )
    )

    return new_jobname


# %%
def cut_binding_sites(df, fasta, window=30, smart_jobname=False):
    """
    Cut the binding sites from the sequences
    Works only for two-chain proteins
    """
    fasta = {jobname: seq for jobname, seq in fasta}

    new_fasta = []
    for row in df.itertuples():
        seq = fasta[row.jobname]
        n_chains = seq.count(":") + 1
        if n_chains != 2:
            logger.warning(f"Skipping {row.jobname} with {n_chains} chains")
            continue

        chain_idx = ord(row.chain2) - ord("A")  # chain2 is the chain to cut
        chain_char = chr(ord("A") + chain_idx)
        # jobname
        if smart_jobname:
            jobnames = row.jobname.split("_")
            chain_jobname = jobnames[chain_idx]
            other_jobname = jobnames[1 - chain_idx]

        # seq
        seqs = seq.split(":")
        chain_seq = seqs[chain_idx]
        other_seq = seqs[1 - chain_idx]

        peaks = list(map(int, row.peaks.split(",")))  # peaks are 0-based and sorted
        for peak in peaks:
            start = max(0, peak - window // 2)
            end = min(len(chain_seq), peak + window // 2 + 1)
            chain_seq_cut = chain_seq[start:end]
            if smart_jobname:
                jobname_cut = cut_jobname(chain_jobname, start, end, seq=chain_seq)
                new_fasta.append(
                    (f"{jobname_cut}_{other_jobname}", f"{chain_seq_cut}:{other_seq}")
                )
            else:
                new_fasta.append(
                    (
                        f"{row.jobname}_{chain_char}_{peak}",
                        f"{chain_seq_cut}:{other_seq}",
                    )
                )

    return new_fasta


# %%
@click.command()
@click.option(
    "-f",
    "--binding-sites-file",
    type=click.Path(dir_okay=False, file_okay=True),
    help="A CSV file containing the binding sites",
    required=True,
)
@click.option(
    "-i",
    "--input",
    type=click.Path(dir_okay=False, file_okay=True),
    help="A fasta file containing the sequences or a results.zip file",
    required=True,
)
@click.option(
    "-o",
    "--output",
    type=click.Path(dir_okay=False, file_okay=True),
    help="Output fasta file",
    required=True,
)
@click.option(
    "--window",
    type=int,
    default=30,
    help="Fragment size = ceil(window / 2) + 1",
    show_default=True,
)
@click.option(
    "--chain",
    type=click.Choice(["A", "B"]),
    help="Chain(s) to cut",
    multiple=True,
    default=["A"],
    show_default=True,
)
@click.option(
    "--smart-jobname",
    help="Modify jobname to reflect the cut. In this case, the jobname should follow the format 'nameA-from-to_nameB-from-to'",
    default=False,
    show_default=True,
)
@click.option("--debug", is_flag=True, help="Print debug information", envvar="DEBUG")
def main(binding_sites_file, input, output, window, chain, smart_jobname, debug):
    # logger level
    logger.setLevel(logging.DEBUG if debug else logging.INFO)

    chains = chain

    df = pd.read_csv(binding_sites_file)
    if input.endswith(".fasta"):
        fasta = read_fasta(input)
    elif input.endswith(".result.zip"):
        seq_csv = read_sequence_csv_file(input)  # ColabFold google colab notebook
        fasta = [(row.id, row.sequence) for row in seq_csv.itertuples()]
    else:
        raise ValueError(f"Unknown input file format {input}")

    fasta_cut = []
    for chain in chains:
        df_chain = df[df["chain2"].eq(chain) & df["chain1"].ne(chain)]
        fasta_cut.extend(cut_binding_sites(df_chain, fasta, window=window))

    write_fasta(fasta_cut, output)


# %%
if __name__ == "__main__":
    main()
