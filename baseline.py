import torch
import tensorflow as tf
import utils
import logging
from metrics import *
import csv

# Settings
flags = tf.compat.v1.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string(
    "target", "curl", "Dataset string."
)  # "curl", "openssl", "sqlite3", "httpd", "libcrypto"
flags.DEFINE_string(
    "embedding_type", "innereye", "Embedding type string."
)  # "innereye", "deepbindiff", "bow"
flags.DEFINE_integer("seed", 10, "Proportion of seeds, 3 means 30%")
flags.DEFINE_string("log", "INFO", "Set log level")

# specify --log=DEBUG or --log=debug
numeric_level = getattr(logging, FLAGS.log.upper(), None)
if not isinstance(numeric_level, int):
    raise ValueError(f"Invalid log level: {FLAGS.log}")
logging.basicConfig(level=numeric_level)

if FLAGS.embedding_type == "innereye":
    FLAGS.seed = 5

# Should use vocab for training
# Load data
(
    adjacency_matrix,
    embeddings_tuple,
    embeddings_mat,
    training_data,
    test_data,
) = utils.load_data(FLAGS.target, FLAGS.embedding_type)

print(f"Baseline (without GCN) : {FLAGS.target}-{FLAGS.embedding_type}")
base_dir_path = utils.get_git_root("train.py") + "/result/"
file_name = base_dir_path + f"baseline-{FLAGS.target}-{FLAGS.embedding_type}.csv"
if FLAGS.embedding_type == "innereye":
    result = get_hits(embeddings_mat.toarray(), test_data)
else:
    result = get_hits(embeddings_mat.toarray(), training_data)
with open(file_name, "w") as csvfile:
    # creating a csv writer object
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(["Hits", "AB", "BA"])
    # writing the data rows
    csvwriter.writerows(result)
