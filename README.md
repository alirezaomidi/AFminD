<div align="center">
<h1 align="center">AFminD: Predicting binding fragments within a big protein</h1>
<!-- <img alt="kaia-llama" height="200px" src="assets/kaia_llama.webp"> -->
<!-- <p align="center"></p> -->


</div>


## Overview


## Getting Started

### Installation
Clone the repository and install the dependencies using [poetry](https://python-poetry.org/):

```bash
poetry install
```

### Data preparation
Each script in the `AFminD.data` module prepares a FASTA file to be submitted to the ColabFold. For example, to prepare the CaV vs. others dataset, run:

```bash
poetry run python -um AFminD.data.prepare_cav_vs_others
```

### minD calculation
```bash
poetry run python -m AFminD.extract_pdist --colabfold-dir /path/to/colabfold/dir/ -o data/cav_vs_others/pdist.csv --n-jobs 10
```

### Binding fragment prediction


### Protein fragmentation