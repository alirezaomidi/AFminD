import requests
import os
from Bio import SeqIO


def download_file(url, output_dir, output_filename=None, overwrite=False):
    assert (
        output_filename is not None or overwrite
    ), "output_filename must be provided if overwrite is False"

    output_path = f"{output_dir}/{output_filename}"
    if not overwrite and os.path.exists(output_path):
        return output_path

    r = requests.get(url)
    if output_filename is None:
        content_disposition = r.headers.get("content-disposition", None)
        if content_disposition:
            output_filename = content_disposition.split("filename=")[1].strip('"')
        else:
            output_filename = url.split("/")[-1]
    output_path = f"{output_dir}/{output_filename}"
    with open(output_path, "wb") as f:
        f.write(r.content)

    return output_path


def read_fasta(fasta_file):
    """
    Read the fasta file and return a dictionary of sequences
    """
    sequences = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        sequences.append((record.description, str(record.seq)))
    return sequences


def write_fasta(fasta, fasta_file):
    """
    Write the fasta file
    """
    with open(fasta_file, "w") as f:
        for name, seq in fasta:
            f.write(f">{name}\n{seq}\n")
