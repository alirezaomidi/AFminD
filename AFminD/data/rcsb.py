import os
import requests
from Bio import SeqIO
import pandas as pd


def download_pdb_fasta(pdb_id, output_dir, overwrite=False):
    output_path = f"{output_dir}/{pdb_id}.fasta"
    if not overwrite and os.path.exists(output_path):
        return output_path
    url = f"https://www.rcsb.org/fasta/entry/{pdb_id}"
    response = requests.get(url)
    response.raise_for_status()
    with open(output_path, "w") as f:
        f.write(response.text)
    return output_path


def download_pdb_mmcif(pdb_id, output_dir, overwrite=False):
    output_path = f"{output_dir}/{pdb_id}.cif"
    if not overwrite and os.path.exists(output_path):
        return output_path
    url = f"https://files.rcsb.org/download/{pdb_id}.cif"
    response = requests.get(url)
    response.raise_for_status()
    with open(output_path, "w") as f:
        f.write(response.text)
    return output_path


def parse_rcsb_fasta(fasta_path):
    chains = {}
    for record in SeqIO.parse(fasta_path, "fasta"):
        chain_id = record.id.split("|")[1].split(":")[-1]
        sequence = str(record.seq)
        chains[chain_id] = sequence
    return chains


def extract_uniprot_alignment(mmcif_dict):
    alignment = {}
    base_key = "_struct_ref_seq"
    for row in zip(
        *[
            mmcif_dict[f"{base_key}.{key}"]
            for key in [
                "pdbx_strand_id",
                "pdbx_db_accession",
                "seq_align_beg",
                "seq_align_end",
                "db_align_beg",
                "db_align_end",
            ]
        ]
    ):
        alignment[row[0]] = tuple(row[1:])
    alignment = pd.DataFrame.from_dict(alignment, orient="index")
    alignment.columns = [
        "uniprot_id",
        "pdb_start",
        "pdb_end",
        "uniprot_start",
        "uniprot_end",
    ]
    alignment.index.name = "chain_id"
    return alignment
