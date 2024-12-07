<div align="center">
<h1 align="center">AFminD: Predicting binding sites within a big protein</h1>
</div>


## Overview
AFminD is a tool that uses AlphaFold-Multimer to predict binding sites within a big protein. AFminD exploits the fact that the distogram data produced by the AlphaFold-Multimer's distogram head encodes potential interactions between residues. By calculating the residue distances between two protein chains, AFminD can pinpoint potential binding sites. The method is described in the following paper:

[Omidi et al., AlphaFold-Multimer accurately captures interactions and dynamics of intrinsically disordered protein regions, Proc. Natl. Acad. Sci. U.S.A., 2024](https://doi.org/10.1073/pnas.2406407121)


## Installation
You need [poetry](https://python-poetry.org/) to be already installed. One way to install it is:
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

Then clone the repository and install the dependencies:

```bash
git clone https://github.com/alirezaomidi/AFminD.git
cd AFminD/
poetry install
```

## How to use

### Run ColabFold
You need to provide both `--zip` and `--save-all` options to ColabFold. The `--save-all` option will save Disogram head's outputs in `.pkl` files, which we use later to calculate minD scores. If you use ColabFold's google colab notebook, simply make sure you have chekced the "save_all" option under "Advanced settings".


### Calculate expected distances from distogram data
The `.pkl` files contain full Distogram data in tensors of shape `N x N x 64`. We need to calculate the expected values for each of the `N x N` probability distributions.
To do so, we use `AFminD.compute_expected_distances` script:
```bash
poetry run python -m AFminD.compute_expected_distances -i /path/to/colabfold/dir/or/zipfile --n-jobs 5
```
Note that the `-i/--input` option can take both a directory containing `.result.zip` files or a single zip file.
The script will calculate expected distances and save them in `.distogram.json` files inside the zip file(s).


### Calculate minD values
Expected distances between each pair of residues are now ready. We can calculate the minimum distance between each residue of one chain and any residue from other chains (a.k.a minD):
```bash
poetry run python -m AFminD.extract_distogram_scores -i /path/to/colabfold/dir/or/zipfile -o minD.csv --n-jobs 10
```

### Predict binding sites
A residues with low minD is part of a potential binding site. To find them, we should normalize minD values and project them inversely to the range $[0, 1]$. Where minD scores peak, a potential binding site is predicted. Use the following script to do the normalization and peak finding:
```bash
poetry run python -m AFminD.find_binding_sites -i minD.csv -o minD_peaks.csv --prominence 0.02 --distance 30
```
The `--prominence` and `--distance` values can be adjusted to your needs. We use $30$ for the distance option to make sure no two peaks are closer than 30 residues, and $0.02$ for the prominence option to make sure no non-significant peaks are found due to noisy signal.


### Fragment proteins
Finally, we can cut the protein of interest to the predicted binding site fragments:
```bash
poetry run python -m AFminD.cut_binding_sites -f minD_peaks.csv --input /path/to/fasta/file/used/for/colabfold.fasta -o fragments.fasta --window 30 --chain A
```
The `--window` option specifies fragment sizes.
The `--chain` option specifies the protein chains to be fragmented. You can specify multiple chains, e.g. `--chain A --chain B`.

From now, you can use the `fragments.fasta` file to run ColabFold again and search for possible boosts in `ipTM` or other metrics shown to be effective in finding potential protein-protein interactions (refer to the [reference](#cite)).


## Cite
```
@article{
    doi:10.1073/pnas.2406407121,
    author = {Alireza Omidi  and Mads Harder Møller  and Nawar Malhis  and Jennifer M. Bui  and Jörg Gsponer },
    title = {AlphaFold-Multimer accurately captures interactions and dynamics of intrinsically disordered protein regions},
    journal = {Proceedings of the National Academy of Sciences},
    volume = {121},
    number = {44},
    pages = {e2406407121},
    year = {2024},
    doi = {10.1073/pnas.2406407121},
    URL = {https://www.pnas.org/doi/abs/10.1073/pnas.2406407121},
    eprint = {https://www.pnas.org/doi/pdf/10.1073/pnas.2406407121},
}
```