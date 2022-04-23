from __future__ import division
from __future__ import print_function

import tensorflow as tf
from utils import *
from metrics import *
from models import GCN_Align
import logging
import os

from xba import XBA

# Initialize torch (stupid method!)
import torch

torch.tensor([1, 2, 3, 4]).to(device="cuda:1")

# Settings
flags = tf.compat.v1.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string(
    "target", "curl", "Dataset string."
)  # "curl", "openssl", "sqlite3", "httpd", "libcrypto", "libc"
flags.DEFINE_string(
    "embedding_type", "innereye", "Embedding type string."
)  # "innereye", "deepbindiff", "bow", "bow-general"
flags.DEFINE_string("test", "curl", "Test program")  # "innereye", "deepbindiff", "bow"

flags.DEFINE_float("learning_rate", 0.0001, "Initial learning rate.")
flags.DEFINE_integer("epochs", 8000, "Number of epochs to train.")
flags.DEFINE_float("dropout", 0.0, "Dropout rate (1 - keep probability).")
flags.DEFINE_float("gamma", 0.1, "Hyper-parameter for margin based loss.")
flags.DEFINE_integer("k", 100, "Number of negative samples for each positive seed.")
flags.DEFINE_integer("ae_dim", 100, "Dimension for AE.")
flags.DEFINE_integer("seed", 3, "Proportion of seeds, 3 means 30%")
flags.DEFINE_string("log", "INFO", "Set log level")
flags.DEFINE_integer("layer", 2, "Number of layers")
flags.DEFINE_bool("record", True, "Record training history")

xba = XBA(
    FLAGS.target,
    FLAGS.embedding_type,
    FLAGS.learning_rate,
    FLAGS.epochs,
    FLAGS.dropout,
    FLAGS.gamma,
    FLAGS.k,
    FLAGS.layer,
    FLAGS.ae_dim,
    FLAGS.seed,
    FLAGS.log,
    FLAGS.record,
    "general-model",
    FLAGS.test,
)

xba.train(restore=True, validate=False)
