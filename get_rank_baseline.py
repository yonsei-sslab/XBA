import tensorflow.compat.v1 as tf
from models import GCN_Align
import utils
import metrics
import logging
import numpy as np
import os
import csv

# Set random seed
seed = 12306
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.compat.v1.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string(
    "target", "curl", "Dataset string."
)  # "curl", "openssl", "sqlite3", "httpd", "libcrypto"
flags.DEFINE_string(
    "embedding_type", "innereye", "Embedding type string."
)  # "innereye", "deepbindiff"
flags.DEFINE_integer("seed", 10, "Proportion of seeds, 3 means 30%")
flags.DEFINE_integer("se_dim", 200, "Dimension for SE.")
flags.DEFINE_integer("ae_dim", 100, "Dimension for AE.")
flags.DEFINE_integer("layer", 2, "Number of layers")
flags.DEFINE_float("dropout", 0.0, "Dropout rate (1 - keep probability).")
flags.DEFINE_float("learning_rate", 20, "Initial learning rate.")
flags.DEFINE_float("gamma", 3.0, "Hyper-parameter for margin based loss.")
flags.DEFINE_integer("k", 5, "Number of negative samples for each positive seed.")
flags.DEFINE_float("beta", 0.9, "Weight for structure embeddings.")
flags.DEFINE_string("log", "INFO", "Set log level")
flags.DEFINE_integer("epochs", 2000, "Number of epochs to train.")
flags.DEFINE_list("bb_id1", [15, 210, 267, 21, 51], "BB ID 1")
flags.DEFINE_list("bb_id2", [163444, 163449, 162301, 166462, 166464], "BB ID 2")
FLAGS.bb_id1 = [int(x) for x in FLAGS.bb_id1]
FLAGS.bb_id2 = [int(x) for x in FLAGS.bb_id2]

# specify --log=DEBUG or --log=debug
numeric_level = getattr(logging, FLAGS.log.upper(), None)
if not isinstance(numeric_level, int):
    raise ValueError(f"Invalid log level: {FLAGS.log}")
logging.basicConfig(level=numeric_level)

# Load data
adjacency_matrix, embeddings_tuple, embeddings_mat, train_data, _ = utils.load_data(
    FLAGS.target, FLAGS.embedding_type
)

print(f"TEST on individual block pairs: Baseline-{FLAGS.target}-{FLAGS.embedding_type}")
result = metrics.get_pair_rank(
    None,
    embeddings_mat.toarray(),
    FLAGS.bb_id1,
    FLAGS.bb_id2,
    0,
)
base_dir_path = utils.get_git_root("train.py") + "/result/"
file_name = (
    base_dir_path
    + f"BBrank-baseline-{FLAGS.target}-{FLAGS.embedding_type}-{FLAGS.bb_id1[:5]}-{FLAGS.bb_id2[:5]}.csv"
)
with open(file_name, "w") as csvfile:
    # creating a csv writer object
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(["BB ID 1", "BB ID 2", "AB", "BA"])
    # writing the data rows
    csvwriter.writerows(result)
