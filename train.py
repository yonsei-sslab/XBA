from __future__ import division
from __future__ import print_function

import tensorflow as tf
from utils import *
from metrics import *
from models import GCN_Align
import logging
import os

# Initialize torch (stupid method!)
import torch

torch.tensor([1, 2, 3, 4]).to(device="cuda:2")

# Set random seed
seed = 12306
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.compat.v1.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string(
    "target", "curl", "Dataset string."
)  # "curl", "openssl", "sqlite3", "httpd", "libcrypto", "libc"
flags.DEFINE_string(
    "embedding_type", "innereye", "Embedding type string."
)  # "innereye", "deepbindiff", "bow"

flags.DEFINE_float("learning_rate", 0.0001, "Initial learning rate.")
flags.DEFINE_integer("epochs", 10000, "Number of epochs to train.")
flags.DEFINE_float("dropout", 0, "Dropout rate (1 - keep probability).")
flags.DEFINE_float("gamma", 0.1, "Hyper-parameter for margin based loss.")
flags.DEFINE_integer("k", 100, "Number of negative samples for each positive seed.")
flags.DEFINE_float("beta", 0.9, "Weight for structure embeddings.")
flags.DEFINE_integer("layer", 2, "Number of layers")
flags.DEFINE_integer("se_dim", 200, "Dimension for SE.")
flags.DEFINE_integer("ae_dim", 200, "Dimension for AE.")
flags.DEFINE_integer("seed", 5, "Proportion of seeds, 3 means 30%")
flags.DEFINE_string("log", "INFO", "Set log level")

record = True

# specify --log=DEBUG or --log=debug
numeric_level = getattr(logging, FLAGS.log.upper(), None)
if not isinstance(numeric_level, int):
    raise ValueError(f"Invalid log level: {FLAGS.log}")
logging.basicConfig(level=numeric_level)

# Load data
adjacency_matrix, embeddings_tuple, embeddings_mat, train_data, test_data = load_data(
    FLAGS.target, FLAGS.embedding_type
)
embeddings_mat = embeddings_mat.toarray()

# Preprocessing
adjacency_matrix_tuple = [preprocess_adj(adjacency_matrix)]
# number of adjacency matrix
num_supports = 1
# We can switch model here
model_func = GCN_Align
num_of_neg_samples = FLAGS.k
num_of_entities = embeddings_tuple[2][0]

tf.compat.v1.disable_v2_behavior()

# Define placeholders
placeholders_attribute_embeddings = {
    "support": [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    "features": tf.placeholder(tf.float32),  # tf.spaese_placeholder(tf.float32),
    "dropout": tf.placeholder_with_default(0.0, shape=()),
    "num_features_nonzero": tf.placeholder_with_default(0, shape=()),
}

# Create model
model_attribute_embeddings = model_func(
    placeholders_attribute_embeddings,
    input_dim=embeddings_tuple[2][1],
    output_dim=FLAGS.ae_dim,
    ILL=train_data,
    sparse_inputs=False,
    featureless=False,
    logging=True,
    num_layer=FLAGS.layer,
)


# Initialize session
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True
# sess = tf.Session(config=config)
sess = tf.Session()

# create saver object
saver = tf.train.Saver()

# Init variables
sess.run(tf.global_variables_initializer())

saved_weights_path = f"./saved_model/tweak-{FLAGS.layer}-{FLAGS.target}-{FLAGS.embedding_type}-seed{FLAGS.seed}0-epoch{FLAGS.epochs}-D{FLAGS.ae_dim}-gamma{FLAGS.gamma}-k{FLAGS.k}-dropout{FLAGS.dropout}-LR{FLAGS.learning_rate}.index"


def plot_history(epoch_history, loss_history, test_history):
    try:
        import matplotlib.pyplot as plt

        plt.style.use("seaborn")  # pretty matplotlib plots

        fig, ax1 = plt.subplots()

        ax1.set_xlabel("epochs")
        ax1.set_ylabel("loss", color="red")
        ax1.plot(
            epoch_history,
            loss_history,
            color="red",
            lw=2,
            ls="-",
            alpha=0.5,
            label="Training Loss",
        )
        ax1.tick_params(axis="y", labelcolor="red")

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        ax2.set_ylabel(
            "hits score", color="blue"
        )  # we already handled the x-label with ax1
        ax2.plot(
            epoch_history,
            test_history,
            color="blue",
            lw=2,
            ls="-",
            alpha=0.5,
            label="Hits score",
        )
        ax2.tick_params(axis="y", labelcolor="blue")

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.savefig(
            f"fig-{FLAGS.target}-{FLAGS.embedding_type}-seed{FLAGS.seed}0-epoch{FLAGS.epochs}-D{FLAGS.ae_dim}-layer{FLAGS.layer}-{FLAGS.target}-{FLAGS.embedding_type}-seed{FLAGS.seed}0-epoch{FLAGS.epochs}-D{FLAGS.ae_dim}-gamma{FLAGS.gamma}-k{FLAGS.k}-dropout{FLAGS.dropout}-LR{FLAGS.learning_rate}.png"
        )

        import json

        history_dict = {
            "epoch": epoch_history,
            "loss": [str(x) for x in loss_history],
            "hits": [str(x) for x in test_history],
        }
        with open(
            f"history-{FLAGS.target}-{FLAGS.embedding_type}-seed{FLAGS.seed}0-epoch{FLAGS.epochs}-D{FLAGS.ae_dim}-layer{FLAGS.layer}.json",
            "w",
        ) as fp:
            json.dump(history_dict, fp)
    except:
        import code

        code.interact(local=locals())


if not os.path.isfile(saved_weights_path):
    # if True:
    train_data_len = len(train_data)
    L = np.ones((train_data_len, num_of_neg_samples)) * (
        train_data[:, 0].reshape((train_data_len, 1))
    )
    # [left_block_id_1, left_block_id_1, left_block_id_1, ..., left_block_id_100, left_block_id_100, left_block_id_100, left_block_id_100, ...]
    negative_samples_left_left = L.reshape((train_data_len * num_of_neg_samples,))
    L = np.ones((train_data_len, num_of_neg_samples)) * (
        train_data[:, 1].reshape((train_data_len, 1))
    )
    # [right_block_id_1, right_block_id_1, right_block_id_1, ..., right_block_id_100, right_block_id_100, right_block_id_100, right_block_id_100, ...]
    negative_samples_right_right = L.reshape((train_data_len * num_of_neg_samples,))

    # Train model
    epoch_history = []
    loss_history = []
    test_history = []
    for epoch in range(FLAGS.epochs):
        if epoch % 10 == 0:
            (
                negative_samples_right_left,
                negative_samples_left_right,
            ) = random_corruption(train_data[:, :2], num_of_neg_samples)

        # Construct feed dictionary
        feed_dict_attribute_embeddings = construct_feed_dict(
            embeddings_mat,
            adjacency_matrix_tuple,
            placeholders_attribute_embeddings,
        )
        feed_dict_attribute_embeddings.update(
            {placeholders_attribute_embeddings["dropout"]: FLAGS.dropout}
        )
        feed_dict_attribute_embeddings.update(
            {
                "neg_left:0": negative_samples_left_left,
                "neg_right:0": negative_samples_left_right,
                "neg2_left:0": negative_samples_right_left,
                "neg2_right:0": negative_samples_right_right,
            }
        )

        # Training step
        outs_attribute_embeddings = sess.run(
            [model_attribute_embeddings.opt_op, model_attribute_embeddings.loss],
            feed_dict=feed_dict_attribute_embeddings,
        )

        # Print results
        logging.info(
            f"Epoch: {epoch + 1:04d} attribute_embeddings_train_loss={outs_attribute_embeddings[1]:.5f}"
        )
        if epoch % 500 == 0 and record:
            epoch_history.append(epoch)
            feed_dict_attribute_embeddings = construct_feed_dict(
                embeddings_mat,
                adjacency_matrix_tuple,
                placeholders_attribute_embeddings,
            )
            vec_ae = sess.run(
                model_attribute_embeddings.outputs,
                feed_dict=feed_dict_attribute_embeddings,
            )
            result = metrics.get_hits(vec_ae, test_data, top_k=[1])
            test_history.append((result[0][1] + result[0][2]) / 2)
            loss_history.append(outs_attribute_embeddings[1])

    # save the variable in the disk
    saved_path = saver.save(
        sess,
        saved_weights_path,
    )
    print(f"model saved in {saved_path}")
    print("Optimization Finished!")

    if record:
        plot_history(epoch_history, loss_history, test_history)
else:
    # restore the saved vairable
    saver.restore(
        sess,
        f"./saved_model/tweak-{FLAGS.target}-{FLAGS.embedding_type}-seed{FLAGS.seed}0-epoch{FLAGS.epochs}-D{FLAGS.ae_dim}-layer{FLAGS.layer}",
    )

# Testing
feed_dict_attribute_embeddings = construct_feed_dict(
    embeddings_mat,
    adjacency_matrix_tuple,
    placeholders_attribute_embeddings,
)
vec_ae = sess.run(
    model_attribute_embeddings.outputs, feed_dict=feed_dict_attribute_embeddings
)

print_and_save_results(test_data, vec_ae)
