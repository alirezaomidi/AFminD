from AFminD.data.utils import read_fasta
from AFminD.data.uniprot import (
    extract_cytoplasmic_regions,
    extract_zinc_fingers,
    download_uniprot_fasta,
)


def _pair_regions(regions1, regions2):
    fasta = [
        (
            f"{region1_name}-{region1_start}-{region1_end}_{region2_name}-{region2_start}-{region2_end}",
            f"{region1_seq}:{region2_seq}",
        )
        for region1_name, region1_start, region1_end, region1_seq in regions1
        for region2_name, region2_start, region2_end, region2_seq in regions2
    ]

    return fasta


def _link_regions(regions, linker_size=50):
    protein_name = regions[0][0]
    assert all(
        protein_name == name for name, _, _, _ in regions
    ), "Regions must be from the same protein"
    regions_concat = ("X" * linker_size).join([region for _, _, _, region in regions])
    regions_concat_name = f"{protein_name}-" + f"-X{linker_size}-".join(
        [f"{start}-{end}" for _, start, end, _ in regions]
    )

    return regions_concat_name, regions_concat


def _pair_linked_regions(regions1, regions2, linker_size=50):
    regions1_concat_name, regions1_concat = _link_regions(
        regions1, linker_size=linker_size
    )
    fasta = [
        (
            f"{regions1_concat_name}_{region2_name}-{region2_start}-{region2_end}",
            f"{regions1_concat}:{region2_seq}",
        )
        for region2_name, region2_start, region2_end, region2_seq in regions2
    ]

    return fasta


def _pair_regions_with_sequence(regions, sequence_name, sequence):
    fasta = [
        (f"{name}-{start}-{end}_{sequence_name}", f"{region}:{sequence}")
        for name, start, end, region in regions
    ]

    return fasta


def _pair_linked_regions_with_sequence(
    regions, sequence_name, sequence, linker_size=50
):
    regions_concat_name, regions_concat = _link_regions(
        regions, linker_size=linker_size
    )
    fasta = [(f"{regions_concat_name}_{sequence_name}", f"{regions_concat}:{sequence}")]

    return fasta


def pair_uniprot_regions_with_zinc_fingers(
    uniprot_id_for_regions,
    uniprot_id_for_zinc_fingers,
    output_dir,
    linker_size=50,
    min_region_length=50,
    max_region_length=None,
    min_zinc_finger_length=20,
    max_zinc_finger_length=None,
):
    # extract regions and zinc fingers
    regions = extract_cytoplasmic_regions(
        uniprot_id_for_regions,
        output_dir,
        min_length=min_region_length,
        max_length=max_region_length,
    )
    zinc_fingers = extract_zinc_fingers(
        uniprot_id_for_zinc_fingers,
        output_dir,
        min_length=min_zinc_finger_length,
        max_length=max_zinc_finger_length,
    )

    # pair regions with zinc fingers
    fasta = _pair_regions(regions, zinc_fingers)
    if len(regions) > 1:
        fasta.extend(
            _pair_linked_regions(
                regions,
                zinc_fingers,
                linker_size=linker_size,
            )
        )

    return fasta


def pair_uniprot_regions_with_uniprot(
    uniprot_id_for_regions,
    uniprot_id,
    output_dir,
    linker_size=50,
    min_length=1,
    max_length=None,
):
    # extract regions and uniprot sequence
    regions = extract_cytoplasmic_regions(
        uniprot_id_for_regions,
        output_dir,
        min_length=min_length,
        max_length=max_length,
    )
    uniprot_sequence_path = download_uniprot_fasta(uniprot_id, output_dir)
    uniprot_sequences = read_fasta(uniprot_sequence_path)
    assert len(uniprot_sequences) == 1, "Uniprot sequence should be a single sequence"
    uniprot_sequence = uniprot_sequences[0][1]

    # pair regions with uniprot
    fasta = _pair_regions_with_sequence(
        regions,
        uniprot_id,
        uniprot_sequence,
    )
    if len(regions) > 1:
        fasta.extend(
            _pair_linked_regions_with_sequence(
                regions, uniprot_id, uniprot_sequence, linker_size=linker_size
            )
        )

    return fasta


def pair_uniprot_regions_with_uniprot_regions(
    uniprot_id_for_regions1,
    uniprot_id_for_regions2,
    output_dir,
    linker_size=50,
    min_length=1,
    max_length=None,
):
    # extract regions
    regions1 = extract_cytoplasmic_regions(
        uniprot_id_for_regions1,
        output_dir,
        min_length=min_length,
        max_length=max_length,
    )
    regions2 = extract_cytoplasmic_regions(
        uniprot_id_for_regions2,
        output_dir,
        min_length=min_length,
        max_length=max_length,
    )

    # pair regions
    fasta = _pair_regions(regions1, regions2)
    if len(regions1) > 1:
        fasta.extend(_pair_linked_regions(regions1, regions2, linker_size=linker_size))

    return fasta
