# XBA
This repository implements XBA, a deep learning tool for generating platform-agnostic binary code embeddings. XBA applies Graph Convolutional Network (GCN) on the graph representation of binary which we call Binary Disassembly Graph (BDG). XBA can learn semantic matchings of binary code compiled for different platfroms that are not included in the training dataset. It outperformed prior works in aligning binary code blocks for different platforms, that shows the embeddings generated by XBA can be used in the cross binary analysis. XBA is implemented with Python v3.8 and Tensorflow v2.7.0. Our GCN implementation is based on this [repository](https://github.com/1049451037/GCN-Align).

![overview](./gcnoverview.jpg)

## Getting Started
### Directory Structure
The working directory should be structured as follows.

    .
    ├── README
    ├── Pipfile                 # Manages a Python virtualenv.
    ├── Pipfile.lock            # Manages a Python virtualenv (Do not touch).
    ├── baseline.py             # Calculate the hit score of baselines (BoW, DeepBinDiff, InnerEye).
    ├── get_rank.py             # Calculate individual rankings of binary code block pairs.
    ├── layers.py               # Define the graph convolution layer.
    ├── metrics.py              # Define the margin-based hinge loss function and hit score calculation.
    ├── models.py               # Define a tensorflow implementation of XBA.
    ├── train_general.py        # Train the model on the whole training dataset across binaries
    ├── train.py                # Train the model.
    ├── utils.py                # Define utility functions.
    ├── xba.py                  # This file defines XBA class that wraps tensorflow specific code for data load, training, validation, test.
    ├── script                  # Includes script files that can reproduce experiments presented in the paper.
    │   ├── table{x}-..         # Reproduce results of the table {x}.
    │   └── test-run.sh         # Test run XBA with 10 epochs for each dataset.
    ├── result                  # Default directory that the hit scores are stored.
    │   └── ...                 #
    ├── history                 # Default directory that the history of training is stored.
    │   └── ...                 #
    ├── data_processing         # 
    │   ├── build_vocab.py      # Build a vocabulary for generating BoW features.
    │   └── split_seed_alignments.py      # Split the training data and test data so that experiment results are deterministic.
    ├── data                    # Graph data for each binary.
    │   ├── curl                # 
    │   │   ├── seed_alignments                # Default directory for split_seed_alignments.py that randomly split alignment.csv into the test data and training data. 
    │   │   ├── alignment.csv                  # Pair-wise-labeled data
    │   │   ├── deepbindiff_embeddings.pt      # DeepBinDiff embeddings for binary code blocks.
    │   │   ├── disasm_innereye.json           # Binary code blocks used to generate InnerEye embeddings.
    │   │   ├── disasm.json                    # Binary code blocks used to generate DeepBinDiff, BoW, and XBA.
    │   │   ├── extfuncs.json                  # 
    │   │   ├── gcn1-relation.csv              # Relation list of graph generated from a binary for Linux
    │   │   ├── gcn2-relation.csv              # Relation list of graph generated from a binary for Windows
    │   │   ├── innereye_embeddings_{curl_openssl_httpd_sqlite3_libcrypto}.pt                # Innereye embedding generated by training on labeled pairs from curl, openssl, httpd, sqlite3, and libcrypto.
    │   │   ├── innereye.csv                   # alignment.csv for training InnerEye (This is not used in XBA)
    │   │   ├── mapping.csv                    #
    │   │   └── subgraph.pth                   #
    │   ├── httpd                              # 
    │   │   └── ...                            # The same structure with curl
    │   ├── libc                               # Graph 1 is generated from a binary for x86 and graph 2 is generated from a binary for arm.
    │   │   └── ...                            # The same structure with curl
    │   ├── libcrypto                          #
    │   │   └── ...                            # The same structure with curl
    │   ├── libcrypto-xarch                    # Graph 1 is generated from a binary for x86 and graph 2 is generated from a binary for arm.
    │   │   └── ...                            # The same structure with curl
    │   ├── openssl                            # 
    │   │   └── ...                            # The same structure with curl
    │   └── sqlite3                            # 
    │       └── ...                            # The same structure with curl
    ├── vocabulary              # Default directory for storing generated vocabulary by build_vocab.py.
    └── saved_model             # Default directory for storing model parameters.  

### Install
#### Prerequisite
Python 3.8 or above version is required. We recommand using Nvidia GPU for training and exhaustive comparison for validation. To install python depedencies, you need to install pipenv first.
```shellsciprt
$ pip3 install pipenv
```

#### Use pipenv shell

Install dependencies
```shellscript
$ pipenv install
```

Activate pipenv shell
```shellscript
$ pipenv shell
```

#### Use your own python virtual environment

Extract requirements.txt
```shellscript
$ pipenv lock -r > requirements.txt
```

Install dependencies
```shellscript
$ pip install -r requirements.txt
```

After activate the python virtual environment, you should be able to run any commands or scripts presented below.

### Dataset
For XBA to learn useful embeddings, software composing our training dataset must have (i) multi-platform support and (ii) platform-specific code blocks. We chose open-source software from the top Github repositories that are widely-used and satisfy the criteria. Selected software covers a broad range of software disciplines; SQLite3 (database), OpenSSL (network), cURL (file transfer), Httpd (web server), libcrypto (crypto library), glibc (standard library). We used IDA pro to extract the graph representation of each binary that is stored in `data` directory.

If users want to run XBA on different binaries, they have to first convert a binary into a proper input format (*i.e.*, Binary Disassembly Graph) that is specified in the paper and structure required files like the above `data` directory structure.

XBA basically requires users to specify the proportion of the dataset to be included in the train split. To make this split deterministic, we first split the dataset and store the train split and test split seperately. Below command will do this for you. You will only need this command if you newly add dataset. As a dafult, we have split our dataset with a ratio of 10/20/30/40/50%.
```shellscript
$ python split_seed_alignments.py --target libcrypto
```
 
### Test run
To test basic funtionality, run the following command.
```shellscript
$ make test
```
It will run the training with BoW and DeepBinDiff base features for each binary in our dataset with 10 epochs and make a validation (approximately it takes around 20 minutes with one RTX 3090). The half of the labeled data is used for training and the another half is for validation. After each training finishes, you will see outputs of hit scores with the validation data as below. Note that the score is not high enough since mostly 10 epochs are not enough to train GCN.
**outputs**
```
...

INFO:root:Epoch: 0001 attribute_embeddings_train_loss=2.56452
INFO:root:Epoch: 0002 attribute_embeddings_train_loss=2.24075
INFO:root:Epoch: 0003 attribute_embeddings_train_loss=1.83440
INFO:root:Epoch: 0004 attribute_embeddings_train_loss=1.37844
INFO:root:Epoch: 0005 attribute_embeddings_train_loss=0.97964
INFO:root:Epoch: 0006 attribute_embeddings_train_loss=0.70114
INFO:root:Epoch: 0007 attribute_embeddings_train_loss=0.53227
INFO:root:Epoch: 0008 attribute_embeddings_train_loss=0.43258
INFO:root:Epoch: 0009 attribute_embeddings_train_loss=0.36502
INFO:root:Epoch: 0010 attribute_embeddings_train_loss=0.31294
INFO:root:model saved in ./saved_model/gcn-5layer-httpd-deepbindiff-seed50-epoch10-D200-gamma3.0-k1-dropout0.0-LR0.001
INFO:root:Optimization Finished!
INFO:root:gcn-5layer-httpd-deepbindiff-seed50-epoch10-D200-gamma3.0-k1-dropout0.0-LR0.001
INFO:root:<Attribute embeddings>
INFO:root:For each left:
INFO:root:Hits@1: 45.76%
INFO:root:Hits@10: 65.14%
INFO:root:Hits@50: 76.32%
INFO:root:Hits@100: 81.33%
INFO:root:For each right:
INFO:root:Hits@1: 47.81%
INFO:root:Hits@10: 66.17%
INFO:root:Hits@50: 77.70%
INFO:root:Hits@100: 82.78%

...
```


## Detailed Description

### XBA class
`xba.py` defines the main class of XBA. It implementes a few core funtionalities of XBA including training, validation, data loading, and building the model. To use XBA, users should fisrt instantiate this class, and by calling proper methods according to the purpose of analysis they can use XBA as it is implemented. If users want to make a specific changes in implementations such as modifying tensorflow training code or changing the model architecture, `xba.py` is a good starting point to take a look.

### Training
The main training script is `train.py` and it needs the following parameters.

* target: A name of target bianry for training. Data will be loaded from `./data/{target}/`. *curl*, *httpd*, *libc*, *libcrypto*, *libcrypto-xarch*, *openssl*, *sqlite3* are available by default.
* embedding_type: A type of base features of binary code blocks that will be an input of the first layer of GCN. BoW and DeepBinDiff are available by default.
* learning_rate: A hyperparameter for the Adam optimizer. The default value is 1e-03.
* epochs: The epoch number of training. Note that the batch size is 1 by default.
* dropout: A hyperparameter for the training. 0 by default.
* gamma: A hyperparameter for the margin-based hinge loss.
* k: The number of negative samples per positive pair.
* layer: The number of GCN layer that XBA uses.
* ae_dim: A dimension of output embedding. By dafult, it is set 200.
* seed: A proportion of the dataset to be included in the train split. For example if it is 3 then 30% of data will be used for training.
* log: Print log if True
* record: Record a history of training in `history` directory.
* restore: Before training, restore the parameters to continue training.
* validate: Record hit scores after training in `result` directory.


Train XBA on {target} based on {embedding_type} with {seed}% seed alignments and store models in `saved_model` directory.

```Python

xba = XBA(
    FLAGS.target, FLAGS.embedding_type, FLAGS.learning_rate, FLAGS.epochs, FLAGS.dropout, FLAGS.gamma,
    FLAGS.k, FLAGS.layer, FLAGS.ae_dim, FLAGS.seed, FLAGS.log, FLAGS.record,
)

(
    adjacency_matrix_tuple,
    embeddings_tuple,
    train_data,
    test_data,
    embeddings_mat,
) = xba.data_load()

placeholders, model = xba.build_model(embeddings_tuple, train_data)

xba.train(
    adjacency_matrix_tuple, embeddings_tuple, train_data, test_data,
    embeddings_mat, placeholders, model, validate=FLAGS.validate, restore=FLAGS.restore,
)
```
If `record` is set to True, then after the training a history of hit scores and the value of loss function is stored in `history` directory. A sample is as follow.
![]()

If `validate` is set to True, then after the training the hit scores using validation data is stored in `result` directory. A sample is as follow.
![]()


### Exaustive Comparisons

### Indivisual Comparisons
Calculate the ranking of indivisual block pairs which did not appeared in seed alignment using fully trained (100% seed alignments) gcn model.
```shellscript
$ pipenv run -- python get_rank.py --target {target} --embedding_type {embedding_type} --log warning --bb_id1="{block id list}" --bb_id2="{block id list}"
```

### Baseline
Run baseline (matching only with BB attribute features)
```shellscript
$ pipenv run -- python baseline.py --target{target} --embedding_type {embedding_type} --log warning --seed {1|2|3|4|5}
```


### Reproduce experiment results
```shellscript
$ make table6
$ make table7
$ make table8
```





## Citation
```
TBA
```