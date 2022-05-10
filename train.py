import tensorflow as tf

from xba import XBA

# Initialize torch (stupid method!)
import torch

torch.tensor([1, 2, 3, 4]).to(device="cuda:2")

# Settings
flags = tf.compat.v1.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string(
    "target", "curl", "Dataset string."
)  # "curl", "openssl", "sqlite3", "httpd", "libcrypto", "libc"
flags.DEFINE_string(
    "embedding_type", "innereye", "Embedding type string."
)  # "innereye", "deepbindiff", "bow"
flags.DEFINE_string("model_name", "", "Saved model file name")
flags.DEFINE_string(
    "test_target",
    "curl",
    "Test program (only required when running the table 8 experiment)",
)
flags.DEFINE_float("learning_rate", 0.001, "Initial learning rate.")
flags.DEFINE_integer("epochs", 2000, "Number of epochs to train.")
flags.DEFINE_float("dropout", 0, "Dropout rate (1 - keep probability).")
flags.DEFINE_float("gamma", 3, "Hyper-parameter for margin based loss.")
flags.DEFINE_integer("k", 100, "Number of negative samples for each positive seed.")
flags.DEFINE_integer("layer", 5, "Number of layers")
flags.DEFINE_integer("ae_dim", 200, "Dimension for AE.")
flags.DEFINE_integer("seed", 5, "Proportion of seeds, 3 means 30%")
flags.DEFINE_string("log", "INFO", "Set log level")
flags.DEFINE_bool("record", True, "Record training history")
flags.DEFINE_bool("restore", False, "Restore and train")
flags.DEFINE_bool("validate", True, "Validate after training")

if not FLAGS.model_name:
    # Model name is not given
    FLAGS.model_name = None

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
    model_file_name=FLAGS.model_name,
    test_program=FLAGS.test_target,
)

(
    adjacency_matrix_tuple,
    embeddings_tuple,
    train_data,
    test_data,
    embeddings_mat,
) = xba.data_load()

(
    placeholders,
    model,
) = xba.build_model(embeddings_tuple, train_data)

xba.train(
    adjacency_matrix_tuple,
    train_data,
    test_data,
    embeddings_mat,
    placeholders,
    model,
    validate=FLAGS.validate,
    restore=FLAGS.restore,
)
