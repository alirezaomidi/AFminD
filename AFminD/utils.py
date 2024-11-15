# %%
import re
import pandas as pd
import zipfile
from Bio.PDB import NeighborSearch
from Bio.PDB.Model import Model
from collections import defaultdict


# %%
def get_jobname(zip_path):
    return re.match(r".*/([a-zA-Z0-9_\-]+)\.result\.zip", zip_path).group(1)


# %%
def parse_jobname_monomer(jobname):
    name, *segments = jobname.split("-")
    # merge each two consequtive non-linker segments
    merged_segments = []
    cur_tuple = None
    for seg in segments:
        if seg.isnumeric():
            if cur_tuple is None:
                cur_tuple = (int(seg),)
            else:
                cur_tuple = cur_tuple + (int(seg),)
                merged_segments.append(cur_tuple)
                cur_tuple = None
        elif seg.startswith("X"):
            # linker
            merged_segments.append(seg)
        else:
            raise ValueError(f"Unexpected segment: {seg}")

    return name, merged_segments


# %%
def parse_jobname(jobname):
    return [parse_jobname_monomer(monomer) for monomer in jobname.split("_")]


# %%
def find_msa_file(zip_path):
    # get jobname
    jobname = get_jobname(zip_path)

    # find all .a3m files in the zip
    msa_filename = None
    with zipfile.ZipFile(zip_path) as z:
        a3m_files = [f for f in z.namelist() if f.endswith(".a3m")]
        msa_filename = f"{jobname}.a3m"
        if msa_filename not in a3m_files:
            msa_filename = f"{jobname}/{jobname}.a3m"  # ColabFold google colab notebook
            if msa_filename not in a3m_files:
                raise ValueError(f"No MSA file found in {zip_path}")

    return pd.DataFrame(
        [(zip_path, msa_filename, jobname)],
        columns=["zip_path", "msa_filename", "jobname"],
    )


# %%
def get_chain_lengths(a3mhead):
    lengths, cardinalities = a3mhead.strip().strip("#").split()
    lengths = list(map(int, lengths.strip().split(",")))
    cardinalities = list(map(int, cardinalities.strip().split(",")))

    chain_lengths = []
    for l, c in zip(lengths, cardinalities):
        chain_lengths.extend([l] * c)

    return chain_lengths


# %%
def format_array(array, fmt="{:.2f}"):
    return ",".join([fmt.format(x) for x in array])


# %%
def find_predictions(zip_path, recycles=False):
    # get jobname
    jobname = get_jobname(zip_path)

    # find pdb files in the zip
    df = []
    with zipfile.ZipFile(zip_path) as z:
        pdb_files = [f for f in z.namelist() if f.endswith(".pdb")]
        for pdb_file in pdb_files:
            pred_jobname, relaxed, rank, model_name, model, seed, recycle = re.match(
                r"(.*)_(un)?relaxed_rank_(\d+)_(.*)_model_(\d+)_seed_(\d+)(?:\.r(\d+))?\.pdb",
                pdb_file,
            ).groups()
            assert pred_jobname == jobname, f'Jobname "{jobname}" does not match regex'
            relaxed = relaxed is None
            rank = int(rank)
            model = int(model)
            seed = int(seed)
            if recycle is not None:
                recycle = int(recycle)
            if not recycles and recycle is not None:
                continue
            df.append(
                (
                    relaxed,
                    rank,
                    model_name,
                    model,
                    seed,
                    recycle,
                    zip_path,
                    pdb_file,
                    jobname,
                )
            )

    return pd.DataFrame(
        df,
        columns=[
            "relaxed",
            "rank",
            "model_name",
            "model",
            "seed",
            "recycle",
            "zip_path",
            "pdb_filename",
            "jobname",
        ],
    )


# %%
def find_interface_residues(
    model: Model, radius=8.0, atom_name="CB", ligand_ids=None, receptor_ids=None
):
    """Map chain to set of interface residues in that chain.

    Args:
        model (Model): BioPython Model object.
        radius (float, optional): Radius to search for interface residues. Defaults to 8.0.

    Returns:
        dict: Dictionary mapping chain IDs to sets of interface residues.
    """
    atoms = [atom for atom in model.get_atoms() if atom.name == atom_name]
    neighbor_search = NeighborSearch(atoms)
    neighbors = neighbor_search.search_all(radius, level="R")

    # keep interactions from different chains
    iface_residues = defaultdict(set)
    for residue1, residue2 in neighbors:
        chain_id1 = residue1.get_parent().get_id()
        chain_id2 = residue2.get_parent().get_id()
        if chain_id1 != chain_id2 and (
            (ligand_ids is None or chain_id1 in ligand_ids)
            and (receptor_ids is None or chain_id2 in receptor_ids)
            or (ligand_ids is None or chain_id2 in ligand_ids)
            and (receptor_ids is None or chain_id1 in receptor_ids)
        ):
            iface_residues[residue1.get_parent().get_id()].add(residue1)
            iface_residues[residue2.get_parent().get_id()].add(residue2)

    return iface_residues
