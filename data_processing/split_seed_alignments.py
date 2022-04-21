from random import seed
from utils import get_git_root, loadfile
import numpy as np
import tensorflow as tf
import os
import csv

# Settings
flags = tf.compat.v1.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string(
    "target", "libcrypto", "Dataset string."
)  # "curl", "openssl", "sqlite3", "httpd", "libcrypto", "libc"

seed_ratio_list = [1, 2, 3, 4, 5]

data_root_dir_path = get_git_root("utils.py") + "/data/done/" + FLAGS.target
reference_alignment_path = data_root_dir_path + "/alignment.csv"

reference_alignment = loadfile(reference_alignment_path, 2)
num_reference_alignment = len(reference_alignment)
np.random.shuffle(reference_alignment)

for seed_ratio in seed_ratio_list:
    train_relations = np.array(
        reference_alignment[: num_reference_alignment // 10 * seed_ratio]
    )
    test_relations = np.array(
        reference_alignment[num_reference_alignment // 10 * seed_ratio :]
    )

    # name of csv file
    seed_dir_name = data_root_dir_path + "/seed_alignments/" + str(seed_ratio) + "/"
    os.makedirs(os.path.dirname(seed_dir_name), exist_ok=True)

    training_alignment = seed_dir_name + "/training_alignments.csv"
    with open(training_alignment, "w") as csvfile:
        # creating a csv writer object
        csvwriter = csv.writer(csvfile)
        # writing the data rows
        csvwriter.writerows(train_relations)

    test_alignment = seed_dir_name + "/test_alignments.csv"
    with open(test_alignment, "w") as csvfile:
        # creating a csv writer object
        csvwriter = csv.writer(csvfile)
        # writing the data rows
        csvwriter.writerows(test_relations)
