import tensorflow.compat.v1 as tf
import numpy as np

import logging
import os
import csv

from utils import (
    load_data,
    preprocess_adj,
    random_corruption,
    construct_feed_dict,
    plot_history,
    print_and_save_results,
)
from models import GCN_Align
import metrics

tf.compat.v1.disable_v2_behavior()


class XBA:
    def __init__(
        self,
        target: str,
        embedding_type: str,
        lr: float,
        epochs: int,
        dropout: float,
        gamma: float,
        k: int,
        layer: int,
        ae_dim: int,
        seed: int,
        log_level: str,
        record: bool,
        model_file_name=None,
        test_program=None,
    ) -> None:
        self.target = target
        self.embedding_type = embedding_type
        self.lr = lr
        self.epochs = epochs
        self.dropout = dropout
        self.gamma = gamma
        self.k = k
        self.layer = layer
        self.ae_dim = ae_dim
        self.seed = seed
        self.record = record
        self.test_program = test_program
        self.model_file_name = model_file_name

        # Set random seed
        np.random.seed(833247)
        tf.set_random_seed(833247)

        # specify --log=DEBUG or --log=debug
        numeric_level = getattr(logging, log_level.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError(f"Invalid log level: {log_level}")
        logging.basicConfig(level=numeric_level)

    def data_load(self):
        (
            adjacency_matrix,
            embeddings_tuple,
            embeddings_mat,
            train_data,
            test_data,
        ) = load_data(self.target, self.embedding_type, self.seed, self.test_program)
        embeddings_mat = embeddings_mat.toarray()

        # Preprocessing
        adjacency_matrix_tuple = [preprocess_adj(adjacency_matrix)]

        return (
            adjacency_matrix_tuple,
            embeddings_tuple,
            train_data,
            test_data,
            embeddings_mat,
        )

    """
    def test(
        self,
        model_dir_path: str = "./saved_model",
        result_dir: str = "./result/",
    ):
        (
            num_supports,
            adjacency_matrix_tuple,
            embeddings_tuple,
            _,
            test_data,
            embeddings_mat,
        ) = self.data_load()

        tf.compat.v1.disable_v2_behavior()

        # Define placeholders
        placeholders_attribute_embeddings = {
            "support": [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
            "features": tf.placeholder(
                tf.float32
            ),  # tf.spaese_placeholder(tf.float32),
            "dropout": tf.placeholder_with_default(0.0, shape=()),
            "num_features_nonzero": tf.placeholder_with_default(0, shape=()),
        }

        # Create model
        model_attribute_embeddings = GCN_Align(
            placeholders_attribute_embeddings,
            input_dim=embeddings_tuple[2][1],
            output_dim=self.ae_dim,
            ILL=test_data,
            sparse_inputs=False,
            featureless=False,
            logging=True,
            num_layer=self.layer,
        )

        # Initialize session
        sess = tf.Session()

        # create saver object
        saver = tf.train.Saver()

        # Init variables
        sess.run(tf.global_variables_initializer())

        # restore the saved vairable
        assert self.model_file_name
        saved_weights_path = os.path.join(
            model_dir_path, self.model_file_name + ".index"
        )
        saver.restore(
            sess,
            saved_weights_path,
        )

        feed_dict_attribute_embeddings = construct_feed_dict(
            embeddings_mat,
            adjacency_matrix_tuple,
            placeholders_attribute_embeddings,
        )
        vec_ae = sess.run(
            model_attribute_embeddings.outputs, feed_dict=feed_dict_attribute_embeddings
        )

        print_and_save_results(
            test_data, vec_ae, self.model_file_name, result_dir, self.record, self.seed
        )
    """

    def exhaustive_comparison(
        self,
        sess,
        adjacency_matrix_tuple,
        embeddings_mat,
        placeholders_attribute_embeddings,
        model_attribute_embeddings,
        bb_id1,
        bb_id2,
        result_dir: str = "./result/",
    ):
        # Testing
        feed_dict_attribute_embeddings = construct_feed_dict(
            embeddings_mat,
            adjacency_matrix_tuple,
            placeholders_attribute_embeddings,
        )
        vec_ae = sess.run(
            model_attribute_embeddings.outputs, feed_dict=feed_dict_attribute_embeddings
        )

        model_file_name, _ = self.get_model_path(" ")

        logging.info(f"TEST on individual block pairs: {model_file_name}")

        result = metrics.get_pair_rank(vec_ae, bb_id1, bb_id2, False)
        file_name = os.path.join(
            result_dir + f"{bb_id1[:5]}-{bb_id2[:5]}" + model_file_name + ".csv"
        )
        with open(file_name, "w") as csvfile:
            # creating a csv writer object
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(["BB ID 1", "BB ID 2", "AB", "BA"])
            # writing the data rows
            csvwriter.writerows(result)

    def restore(self, model_dir_path: str = "./saved_model"):
        # Initialize session
        sess = tf.Session()

        # create saver object
        saver = tf.train.Saver()

        # Init variables
        sess.run(tf.global_variables_initializer())

        model_file_name, saved_weights_path = self.get_model_path(model_dir_path)

        if os.path.isfile(saved_weights_path):
            # restore the saved vairable
            saver.restore(sess, saved_weights_path)

        return sess

    def train(
        self,
        adjacency_matrix_tuple,
        embeddings_tuple,
        train_data,
        test_data,
        embeddings_mat,
        placeholders_attribute_embeddings,
        model_attribute_embeddings,
        history_dir_path: str = "./history",
        model_dir_path: str = "./saved_model",
        restore: bool = False,
        validate: bool = True,
    ):

        # Initialize session
        sess = tf.Session()

        # create saver object
        saver = tf.train.Saver()

        # Init variables
        sess.run(tf.global_variables_initializer())

        model_file_name, saved_weights_path = self.get_model_path(model_dir_path)

        if restore:
            if os.path.isfile(saved_weights_path + ".index"):
                # restore the saved vairable
                saver.restore(sess, saved_weights_path)

        # Generate negative samples
        train_data_len = len(train_data)
        L = np.ones((train_data_len, self.k)) * (
            train_data[:, 0].reshape((train_data_len, 1))
        )
        # [left_block_id_1, left_block_id_1, left_block_id_1, ..., left_block_id_100, left_block_id_100, left_block_id_100, left_block_id_100, ...]
        negative_samples_left_left = L.reshape((train_data_len * self.k,))
        L = np.ones((train_data_len, self.k)) * (
            train_data[:, 1].reshape((train_data_len, 1))
        )
        # [right_block_id_1, right_block_id_1, right_block_id_1, ..., right_block_id_100, right_block_id_100, right_block_id_100, right_block_id_100, ...]
        negative_samples_right_right = L.reshape((train_data_len * self.k,))

        # Train model
        epoch_history = []
        loss_history = []
        test_history = []
        for epoch in range(self.epochs):
            if epoch % 100 == 0:
                (
                    negative_samples_right_left,
                    negative_samples_left_right,
                ) = random_corruption(train_data[:, :2], self.k)

            # Construct feed dictionary
            feed_dict_attribute_embeddings = construct_feed_dict(
                embeddings_mat,
                adjacency_matrix_tuple,
                placeholders_attribute_embeddings,
            )
            feed_dict_attribute_embeddings.update(
                {placeholders_attribute_embeddings["dropout"]: self.dropout}
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
            if epoch % 500 == 0 and self.record:
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
        logging.info(f"model saved in {saved_path}")
        logging.info("Optimization Finished!")

        if self.record:
            plot_history(
                epoch_history,
                loss_history,
                test_history,
                history_dir_path,
                fig_name=model_file_name,
            )
        if validate:
            self.validate(
                sess,
                model_attribute_embeddings,
                test_data,
                embeddings_mat,
                adjacency_matrix_tuple,
                placeholders_attribute_embeddings,
                model_file_name,
            )

    def build_model(self, embeddings_tuple, train_data):
        # Define placeholders
        placeholders_attribute_embeddings = {
            "support": [tf.sparse_placeholder(tf.float32) for _ in range(1)],
            "features": tf.placeholder(
                tf.float32
            ),  # tf.spaese_placeholder(tf.float32),
            "dropout": tf.placeholder_with_default(0.0, shape=()),
            "num_features_nonzero": tf.placeholder_with_default(0, shape=()),
        }

        # Create model
        model_attribute_embeddings = GCN_Align(
            placeholders_attribute_embeddings,
            input_dim=embeddings_tuple[2][1],
            output_dim=self.ae_dim,
            ILL=train_data,
            sparse_inputs=False,
            featureless=False,
            logging=True,
            num_layer=self.layer,
        )

        return placeholders_attribute_embeddings, model_attribute_embeddings

    def get_model_path(self, model_dir_path):
        if self.model_file_name is None:
            model_file_name = f"gcn-{self.layer}layer-{self.target}-{self.embedding_type}-seed{self.seed}0-epoch{self.epochs}-D{self.ae_dim}-gamma{self.gamma}-k{self.k}-dropout{self.dropout}-LR{self.lr}"
        else:
            model_file_name = self.model_file_name
        saved_weights_path = os.path.join(model_dir_path, model_file_name)

        return model_file_name, saved_weights_path

    def validate(
        self,
        sess,
        model_attribute_embeddings,
        test_data,
        embeddings_mat,
        adjacency_matrix_tuple,
        placeholders_attribute_embeddings,
        model_file_name,
        result_dir: str = "./result/",
    ):
        feed_dict_attribute_embeddings = construct_feed_dict(
            embeddings_mat,
            adjacency_matrix_tuple,
            placeholders_attribute_embeddings,
        )
        vec_ae = sess.run(
            model_attribute_embeddings.outputs, feed_dict=feed_dict_attribute_embeddings
        )

        print_and_save_results(
            test_data, vec_ae, model_file_name, result_dir, self.record, self.seed
        )
