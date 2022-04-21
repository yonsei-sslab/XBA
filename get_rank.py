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
flags.DEFINE_float("learning_rate", 0.0001, "Initial learning rate.")
flags.DEFINE_float("gamma", 0.1, "Hyper-parameter for margin based loss.")
flags.DEFINE_integer("k", 10, "Number of negative samples for each positive seed.")
flags.DEFINE_float("beta", 0.9, "Weight for structure embeddings.")
flags.DEFINE_string("log", "INFO", "Set log level")
flags.DEFINE_integer("epochs", 35000, "Number of epochs to train.")
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
embeddings_mat = embeddings_mat.toarray()

# number of adjacency matrix
num_supports = 1
# We can switch model here
model_func = GCN_Align
num_of_entities = embeddings_tuple[2][0]
num_of_neg_samples = FLAGS.k
# Preprocessing
adjacency_matrix_tuple = [utils.preprocess_adj(adjacency_matrix)]

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

# create saver object
saver = tf.train.Saver()

# run the session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    saved_weights_path = f"./saved_model/tweak-layer{FLAGS.layer}-{FLAGS.target}-{FLAGS.embedding_type}-seed{FLAGS.seed}0-epoch{FLAGS.epochs}-D{FLAGS.ae_dim}.index"
    if not os.path.isfile(saved_weights_path):
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
        # Init variables
        sess.run(tf.global_variables_initializer())

        # Train model
        for epoch in range(FLAGS.epochs):
            if epoch % 10 == 0:
                (
                    negative_samples_right_left,
                    negative_samples_left_right,
                ) = utils.random_corruption(train_data[:, :2], num_of_neg_samples)

            # Construct feed dictionary
            feed_dict_attribute_embeddings = utils.construct_feed_dict(
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

        # save the variable in the disk
        saved_path = saver.save(
            sess,
            f"./saved_model/tweak-layer{FLAGS.layer}-{FLAGS.target}-{FLAGS.embedding_type}-seed{FLAGS.seed}0-epoch{FLAGS.epochs}-D{FLAGS.ae_dim}",
        )
        print(f"model saved in {saved_path}")
        print("Optimization Finished!")
    else:
        # restore the saved vairable
        saver.restore(
            sess,
            f"./saved_model/tweak-layer{FLAGS.layer}-{FLAGS.target}-{FLAGS.embedding_type}-seed{FLAGS.seed}0-epoch{FLAGS.epochs}-D{FLAGS.ae_dim}",
        )

    # Testing
    feed_dict_attribute_embeddings = utils.construct_feed_dict(
        embeddings_mat,
        adjacency_matrix_tuple,
        placeholders_attribute_embeddings,
    )
    vec_ae = sess.run(
        model_attribute_embeddings.outputs, feed_dict=feed_dict_attribute_embeddings
    )

    print(
        f"TEST on individual block pairs: GCN-{len(model_attribute_embeddings.layers)}layers-{FLAGS.target}-{FLAGS.embedding_type}-seed{FLAGS.seed}0"
    )
    result = metrics.get_pair_rank(
        None, vec_ae, FLAGS.bb_id1, FLAGS.bb_id2, FLAGS.beta, False
    )
    base_dir_path = utils.get_git_root("train.py") + "/result/"
    file_name = (
        base_dir_path
        + f"BBrank-GCN-{len(model_attribute_embeddings.layers)}layers-{FLAGS.target}-{FLAGS.embedding_type}-{FLAGS.bb_id1[:5]}-{FLAGS.bb_id2[:5]}.csv"
    )
    with open(file_name, "w") as csvfile:
        # creating a csv writer object
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(["BB ID 1", "BB ID 2", "AB", "BA"])
        # writing the data rows
        csvwriter.writerows(result)
