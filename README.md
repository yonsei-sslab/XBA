# XBA
This repository implements XBA, a deep learning tool for generating platform-agnostic binary code embeddings. XBA takes a semi-supervised approach to learn semantic matching of binary code blocks compiled for different platforms by using graph neural network with siamese architecture. It outperformed prior works

## Setup

### Structure

    .
    ├── ...
    ├── README
    ├── Pipfile                 # Manages a Python virtualenv
    ├── Pipfile.lock            # Manages a Python virtualenv (Do not touch)
    ├── test                    # Test files (alternatively `spec` or `tests`)
    │   ├── benchmarks          # Load and stress tests
    │   ├── integration         # End-to-end, integration tests (alternatively `e2e`)
    │   └── unit                # Unit tests
    └── ...

### Install

### Dataset




## Getting Started

## Detailed Description

### Training

### Exaustive Comparisons

### Indivisual Comparisons



### data processing

## How to run
Prerequisites
```shellsciprt
$ pip3 install pipenv
```

Install dependencies
```shellscript
$ pipenv install
```

Training GCN model on {target} based on {embedding_type} with 10/20/30/40/50% seed alignments and store models in `saved_model`. Results will be stored in `result` directory in root directory.
```shellscript
$ pipenv run -- python train.py --target {target} --embedding_type {embedding_type} --seed {1|2|3|4|5} --log warning
```

Run baseline (matching only with BB attribute features)
```shellscript
$ pipenv run -- python baseline.py --target{target} --embedding_type {embedding_type} --log warning --seed {1|2|3|4|5}
```

Calculate the ranking of indivisual block pairs which did not appeared in seed alignment using fully trained (100% seed alignments) gcn model.
```shellscript
$ pipenv run -- python get_rank.py --target {target} --embedding_type {embedding_type} --log warning --bb_id1="{block id list}" --bb_id2="{block id list}"
```

Split alignments data into training and test data with ratio of 10/20/30/40/50%
```shellscript
$ python split_seed_alignments.py --target libcrypto
```

## Citation
```
TBA
```