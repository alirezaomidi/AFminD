import json
import os
import yaml
from Bio import SeqIO
from AFminD.data.utils import download_file


def download_uniprot_fasta(uniprot_id, output_dir, overwrite=False):
    fasta_url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
    return download_file(
        fasta_url, output_dir, f"{uniprot_id}.fasta", overwrite=overwrite
    )


def download_uniprot_annot(uniprot_id, output_dir, overwrite=False):
    fields = [
        "ft_var_seq",
        "ft_variant",
        "ft_non_cons",
        "ft_non_std",
        "ft_non_ter",
        "ft_conflict",
        "ft_unsure",
        "ft_act_site",
        "ft_binding",
        "ft_dna_bind",
        "ft_site",
        "ft_mutagen",
        "ft_intramem",
        "ft_topo_dom",
        "ft_transmem",
        "ft_chain",
        "ft_crosslnk",
        "ft_disulfid",
        "ft_carbohyd",
        "ft_init_met",
        "ft_lipid",
        "ft_mod_res",
        "ft_peptide",
        "ft_propep",
        "ft_signal",
        "ft_transit",
        "ft_strand",
        "ft_helix",
        "ft_turn",
        "ft_coiled",
        "ft_compbias",
        "ft_domain",
        "ft_motif",
        "ft_region",
        "ft_repeat",
        "ft_zn_fing",
    ]
    annot_url = (
        f"https://www.uniprot.org/uniprot/{uniprot_id}.json?fields={','.join(fields)}"
    )
    return download_file(
        annot_url, output_dir, f"{uniprot_id}.json", overwrite=overwrite
    )


def _parse_uniprot_fasta(fasta_path):
    record = SeqIO.read(fasta_path, "fasta")
    name = record.description.split("|")[1]
    sequence = str(record.seq)
    return name, sequence


def _load_custom_features(
    uniprot_id,
    custom_features_path="custom_features.yml",
):
    with open(os.path.join(os.path.dirname(__file__), custom_features_path), "r") as f:
        custom_features = yaml.safe_load(f)
    # filter custom regions to only include the given uniprot_id
    custom_features = {k: v for k, v in custom_features.items() if k == uniprot_id}
    assert (
        len(custom_features) <= 1
    ), f"Expected at most 1 custom entry for {uniprot_id}, but found {len(custom_features)}"

    return custom_features.get(uniprot_id, None)


def _extract_regions(
    uniprot_fasta_path,
    uniprot_annot_path,
    region_type,
    region_desc=None,
    min_length=1,
    max_length=None,
    custom_features=None,
):
    name, sequence = _parse_uniprot_fasta(uniprot_fasta_path)
    with open(uniprot_annot_path, "r") as f:
        annot = json.load(f)

    if custom_features is not None:
        # merge custom features with annot
        annot["features"].extend(custom_features["features"])

    regions = []
    for feature in annot["features"]:
        if feature["type"] != region_type or (
            region_desc is not None and feature["description"] != region_desc
        ):
            continue
        start = feature["location"]["start"]["value"]
        end = feature["location"]["end"]["value"]
        region = sequence[start - 1 : end]
        if len(region) < min_length or (
            max_length is not None and len(region) > max_length
        ):
            continue
        regions.append((name, start, end, region))
    return regions


def extract_cytoplasmic_regions(
    uniprot_id,
    output_dir,
    min_length=50,
    max_length=None,
):
    uniprot_fasta_path = download_uniprot_fasta(uniprot_id, output_dir)
    uniprot_annot_path = download_uniprot_annot(uniprot_id, output_dir)
    custom_features = _load_custom_features(uniprot_id)

    return _extract_regions(
        uniprot_fasta_path,
        uniprot_annot_path,
        "Topological domain",
        "Cytoplasmic",
        min_length=min_length,
        max_length=max_length,
        custom_features=custom_features,
    )


def extract_zinc_fingers(
    uniprot_id,
    output_dir,
    min_length=20,
    max_length=None,
):
    uniprot_fasta_path = download_uniprot_fasta(uniprot_id, output_dir)
    uniprot_annot_path = download_uniprot_annot(uniprot_id, output_dir)
    custom_features = _load_custom_features(uniprot_id)

    return _extract_regions(
        uniprot_fasta_path,
        uniprot_annot_path,
        "Zinc finger",
        min_length=min_length,
        max_length=max_length,
        custom_features=custom_features,
    )
