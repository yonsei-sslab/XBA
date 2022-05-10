import tensorflow.compat.v1 as tf
import numpy as np
import scipy
import scipy.spatial
import torch
import logging

flags = tf.compat.v1.app.flags
FLAGS = flags.FLAGS


def get_placeholder_by_name(name):
    try:
        return tf.get_default_graph().get_tensor_by_name(name + ":0")
    except:
        return tf.placeholder(tf.int32, name=name)


def align_loss(outlayer, ILL, gamma, k):
    left = ILL[:, 0]
    right = ILL[:, 1]
    t = len(ILL)
    left_x = tf.nn.embedding_lookup(outlayer, left)
    right_x = tf.nn.embedding_lookup(outlayer, right)
    A = tf.reduce_sum(tf.abs(left_x - right_x), 1)
    neg_left = get_placeholder_by_name(
        "neg_left"
    )  # tf.placeholder(tf.int32, [t * k], "neg_left")
    neg_right = get_placeholder_by_name(
        "neg_right"
    )  # tf.placeholder(tf.int32, [t * k], "neg_right")
    neg_l_x = tf.nn.embedding_lookup(outlayer, neg_left)
    neg_r_x = tf.nn.embedding_lookup(outlayer, neg_right)
    B = tf.reduce_sum(tf.abs(neg_l_x - neg_r_x), 1)
    C = -tf.reshape(B, [t, k])
    D = A + gamma
    L1 = tf.nn.relu(tf.add(C, tf.reshape(D, [t, 1])))
    neg_left = get_placeholder_by_name(
        "neg2_left"
    )  # tf.placeholder(tf.int32, [t * k], "neg2_left")
    neg_right = get_placeholder_by_name(
        "neg2_right"
    )  # tf.placeholder(tf.int32, [t * k], "neg2_right")
    neg_l_x = tf.nn.embedding_lookup(outlayer, neg_left)
    neg_r_x = tf.nn.embedding_lookup(outlayer, neg_right)
    B = tf.reduce_sum(tf.abs(neg_l_x - neg_r_x), 1)
    C = -tf.reshape(B, [t, k])
    L2 = tf.nn.relu(tf.add(C, tf.reshape(D, [t, 1])))
    return (tf.reduce_sum(L1) + tf.reduce_sum(L2)) / (2.0 * k * t)


def get_pair_rank(
    ae_vec: list,
    bb_id1: list,
    bb_id2: list,
):
    test_data1, test_data2 = get_test_data()
    vec = ae_vec

    Lvec = np.array([vec[l] for l in test_data1])
    Rvec = np.array([vec[r] for r in test_data2])

    result = []
    for _, (bb1, bb2) in enumerate(zip(bb_id1, bb_id2)):
        distance = scipy.spatial.distance.cityblock(vec[bb1], vec[bb2])

        sim = [scipy.spatial.distance.cityblock(vec[bb1], y) for y in Rvec]

        rank_index_lr = sum(sim < distance)
        print(f"For BB-{bb1}, BB-{bb2} ranked {rank_index_lr} among {len(Rvec)}")

        sim = [scipy.spatial.distance.cityblock(vec[bb2], y) for y in Lvec]

        rank_index_rl = sum(sim < distance)
        print(f"For BB-{bb2}, BB-{bb1} ranked {rank_index_rl} among {len(Lvec)}")

        result.append([bb1, bb2, rank_index_lr, rank_index_rl])

    return result


def get_test_data():
    from utils import get_git_root, load_relation_file

    data_root_dir_path = get_git_root("utils.py") + "/data/done/" + FLAGS.target
    relation_triples_path_left = data_root_dir_path + "/gcn1-relation.csv"
    relation_triples_path_right = data_root_dir_path + "/gcn2-relation.csv"
    relation_set1 = load_relation_file(relation_triples_path_left)
    relation_set2 = load_relation_file(relation_triples_path_right)
    test_data1 = set([relation[i] for i in [0, 2] for relation in relation_set1])
    test_data2 = set([relation[i] for i in [0, 2] for relation in relation_set2])
    return test_data1, test_data2


def get_rank_idx(i: int, sim: list):
    element = sim[i]

    return torch.count_nonzero(sim < element)


def get_hits(vec, test_pair, top_k=(1, 10, 50, 100)):
    Lvec = np.array([vec[e1].astype(float) for e1, _ in test_pair])
    Rvec = np.array([vec[e2].astype(float) for _, e2 in test_pair])

    Lvec_tensor = torch.tensor(Lvec)
    Rvec_tensor = torch.tensor(Rvec)
    sim = torch.cdist(Lvec_tensor, Rvec_tensor, 1)

    try:
        sim = sim.to(device="cuda:1")
    except:
        pass

    top_lr = [0] * len(top_k)
    for i in range(Lvec.shape[0]):
        rank_index = get_rank_idx(i, sim[i, :])
        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                top_lr[j] += 1

    top_rl = [0] * len(top_k)
    for i in range(Rvec.shape[0]):
        rank_index = get_rank_idx(i, sim[:, i])
        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                top_rl[j] += 1

    logging.info("For each left:")
    result = []
    for i in range(len(top_lr)):
        logging.info("Hits@%d: %.2f%%" % (top_k[i], top_lr[i] / len(test_pair) * 100))
        result.append([top_k[i], top_lr[i] / len(test_pair) * 100])
    logging.info("For each right:")
    for i in range(len(top_rl)):
        logging.info("Hits@%d: %.2f%%" % (top_k[i], top_rl[i] / len(test_pair) * 100))
        result[i].append(top_rl[i] / len(test_pair) * 100)

    return result
