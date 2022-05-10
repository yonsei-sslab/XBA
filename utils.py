import logging
import numpy as np
import scipy.sparse as sp
import git
import json
import torch
import csv
import metrics
import pandas as pd

import os


def plot_history(
    epoch_history,
    loss_history,
    test_history,
    dir_path: str = "history",
    fig_name: str = "loss_history",
):
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

    plt.savefig(os.path.join(dir_path, fig_name + ".png"))

    import json

    history_dict = {
        "epoch": epoch_history,
        "loss": [str(x) for x in loss_history],
        "hits": [str(x) for x in test_history],
    }
    with open(
        os.path.join(dir_path, fig_name + ".json"),
        "w",
    ) as fp:
        json.dump(history_dict, fp)


def print_and_save_results(
    test_data, vec_ae, model_name: str, result_dir: str, record: bool, seed: int
):
    logging.info(model_name)
    logging.info("<Attribute embeddings>")

    result = metrics.get_hits(vec_ae, test_data)
    if record:
        file_name = os.path.join(result_dir + model_name + ".csv")
        with open(file_name, "w") as csvfile:
            # creating a csv writer object
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(["Proportion", "Hits", "AB", "BA"])
            # writing the data rows
            result = [[seed * 10] + x for x in result]
            csvwriter.writerows(result)


def random_corruption(alignments: np.ndarray, num_of_neg_samples: int) -> tuple:
    left_right = []
    right_left = []
    for i, _ in enumerate(alignments):
        indexing = np.ones((alignments.shape[0],), dtype=bool)
        indexing[i] = False
        left_right_candidate = np.random.choice(
            alignments[indexing, 1].flatten(), num_of_neg_samples
        )
        right_left_candidate = np.random.choice(
            alignments[indexing, 0].flatten(), num_of_neg_samples
        )

        left_right = np.hstack((left_right, left_right_candidate))
        right_left = np.hstack((right_left, right_left_candidate))

    return left_right, right_left


def get_git_root(path):
    git_repo = git.Repo(path, search_parent_directories=True)
    git_root = git_repo.git.rev_parse("--show-toplevel")

    return git_root


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""

    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    return to_tuple(sparse_mx)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def construct_feed_dict(features, support, placeholders):
    """Construct feed dictionary for GCN-Align."""
    feed_dict = dict()
    feed_dict.update({placeholders["features"]: features})
    feed_dict.update(
        {placeholders["support"][i]: support[i] for i in range(len(support))}
    )
    return feed_dict


def loadfile(fn, num=1):
    """Load a file and return a list of tuple containing $num integers in each line."""
    logging.info("loading a file..." + fn)
    ret = []
    with open(fn, encoding="utf-8") as f:
        for line in f:
            th = line[:-1].split(",")
            x = []
            for i in range(num):
                x.append(int(th[i]))
            ret.append(tuple(x))
    return ret


def load_relation_file(fn):
    """Load a file and return a list of tuple containing $num integers in each line."""
    logging.info("loading a file..." + fn)
    ret = []
    relation2id = {}
    counter = 0
    with open(fn, encoding="utf-8") as f:
        for line in f:
            th = line[:-1].split(",")
            x = []
            x.append(int(th[0]))
            if th[1] in relation2id:
                x.append(relation2id.get(th[1]))
            else:
                relation2id[th[1]] = counter
                counter += 1
                x.append(relation2id.get(th[1]))
            x.append(int(th[2]))
            ret.append(tuple(x))

    return ret


def loadattr(embedding_path, num_of_entities, embe_type):
    embeddings = torch.load(embedding_path)

    # Check if we have a correct embeddings
    if embe_type == "innereye":
        assert (
            embeddings.shape[0] == num_of_entities
        ), f"{embeddings.shape[0]} != {num_of_entities}"
        assert embeddings.shape[1] == 50, f"{embeddings.shape[1]} != 50"
    elif embe_type == "deepbindiff":
        assert (
            embeddings.shape[0] == num_of_entities
        ), f"{embeddings.shape[0]} != {num_of_entities}"
        assert embeddings.shape[1] == 128, f"{embeddings.shape[1]} != 128"

    # build attribute feature matrix
    return sp.coo_matrix(embeddings.numpy())


def func(KG):
    head = {}
    cnt = {}
    for tri in KG:
        if tri[1] not in cnt:
            cnt[tri[1]] = 1
            head[tri[1]] = set([tri[0]])
        else:
            cnt[tri[1]] += 1
            head[tri[1]].add(tri[0])
    r2f = {}
    for r in cnt:
        r2f[r] = len(head[r]) / cnt[r]

    # Ablation study for relations
    # r2f[1] = 0

    return r2f


def ifunc(KG):
    tail = {}
    cnt = {}
    for tri in KG:
        if tri[1] not in cnt:
            cnt[tri[1]] = 1
            tail[tri[1]] = set([tri[2]])
        else:
            cnt[tri[1]] += 1
            tail[tri[1]].add(tri[2])
    r2if = {}
    for r in cnt:
        r2if[r] = len(tail[r]) / cnt[r]

    # Ablation study for relations
    # r2if[1] = 0

    return r2if


def get_weighted_adj(num_of_entities, relation_set):
    # Functionality
    r2f = func(relation_set)
    # Inverse functionality
    r2if = ifunc(relation_set)

    M = {}
    for tri in relation_set:
        # self-looping relation
        if tri[0] == tri[2]:
            continue

        # Add inverse functionality score with the maximum value 0.3
        if (tri[0], tri[2]) not in M:
            M[(tri[0], tri[2])] = max(r2if[tri[1]], 0.0)
        else:
            M[(tri[0], tri[2])] += max(r2if[tri[1]], 0.0)

        # Add functionality score with the maximum value 0.3
        if (tri[2], tri[0]) not in M:
            M[(tri[2], tri[0])] = max(r2f[tri[1]], 0.0)
        else:
            M[(tri[2], tri[0])] += max(r2f[tri[1]], 0.0)

    row = []
    col = []
    data = []
    for key in M:
        row.append(key[1])
        col.append(key[0])
        data.append(M[key])

    return sp.coo_matrix((data, (row, col)), shape=(num_of_entities, num_of_entities))


def load_data(target: str, embe_type: str, seed: int, test_program=None):
    data_root_dir_path = get_git_root("utils.py") + "/data/done/" + target
    entity_id_path = data_root_dir_path + "/disasm.json"
    if embe_type == "innereye":
        if target in ["curl", "openssl", "libcrypto", "sqlite3", "httpd"]:
            embeddings_path = (
                data_root_dir_path
                + "/"
                + embe_type
                + "_embeddings_curl_openssl_httpd_sqlite3_libcrypto_5.pt"
            )
        else:
            embeddings_path = (
                data_root_dir_path
                + "/"
                + embe_type
                + "_embeddings_libcrypto-xarch_libc_5.pt"
            )
    elif embe_type == "deepbindiff":
        embeddings_path = data_root_dir_path + "/" + embe_type + "_embeddings.pt"
    elif embe_type == "bow":
        pass
    relation_triples_path_left = data_root_dir_path + "/gcn1-relation.csv"
    relation_triples_path_right = data_root_dir_path + "/gcn2-relation.csv"
    reference_alignment_path = data_root_dir_path + "/seed_alignments/" + str(seed)

    f = open(entity_id_path, "r")
    id2entity = json.load(f)
    num_of_entities = max([int(key) for key in id2entity.keys()]) + 1

    # Training and test dataset split
    if seed == 10:
        train_relations = np.array(loadfile(data_root_dir_path + "/alignment.csv", 2))
        test_relations = []
    elif seed == 0:
        test_relations = np.array(loadfile(data_root_dir_path + "/alignment.csv", 2))
        train_relations = []
    else:
        train_relations = np.array(
            loadfile(reference_alignment_path + "/training_alignments.csv", 2)
        )
        test_relations = np.array(
            loadfile(reference_alignment_path + "/test_alignments.csv", 2)
        )

    # Load relation set
    relation_set = load_relation_file(relation_triples_path_left) + load_relation_file(
        relation_triples_path_right
    )

    if embe_type == "bow":
        disasm_path = data_root_dir_path + "/disasm_innereye.json"
        if seed == 10:
            vocab_path = f"./vocabulary/{target}_vocabulary_for_new_alignments.json"
        else:
            vocab_path = f"./vocabulary/{target}_vocabulary.json"
        attribute_mat = load_bow_embeddings(
            disasm_path, vocab_path, target, embe_type, test_program
        )
        attribute_tuple = sparse_to_tuple(sp.coo_matrix(attribute_mat))
    elif embe_type == "bow-general":
        disasm_path = data_root_dir_path + "/disasm_innereye.json"
        program_list = [
            "curl",
            "openssl",
            "httpd",
            "sqlite3",
            "libcrypto",
        ]

        assert test_program is not None

        program_list = [e for e in program_list if e != test_program]
        vocab_path = f'./vocabulary/{"-".join(program_list)}_vocabulary.json'
        attribute_mat = load_bow_embeddings(
            disasm_path, vocab_path, target, embe_type, test_program
        )
        attribute_tuple = sparse_to_tuple(sp.coo_matrix(attribute_mat))
    else:
        attribute_mat = loadattr(embeddings_path, num_of_entities, embe_type)
        # COO representation of attribute__mat
        attribute_tuple = sparse_to_tuple(sp.coo_matrix(attribute_mat))
    adjacency_mat = get_weighted_adj(num_of_entities, relation_set)

    return (
        adjacency_mat,
        attribute_tuple,
        attribute_mat,
        train_relations,
        test_relations,
    )


def load_bow_embeddings(
    disasm_path: str, vocab_path: str, target: str, embedding_type: str, test: str
):
    f = open(disasm_path, "r")
    bb_list = json.load(f)
    f = open(vocab_path, "r")
    vocab = json.load(f)
    num_of_entities = max([int(key) for key in bb_list.keys()]) + 1

    data_root_dir_path = get_git_root("utils.py") + "/data/done/" + target
    seed_path = data_root_dir_path + "/alignment.csv"
    alignments = pd.read_csv(seed_path, header=None)
    alignments = list(alignments[0]) + list(alignments[1])

    row = []
    col = []
    data = []
    stat = {}
    for idx in range(num_of_entities):
        bb_contents = bb_list.get(str(idx), [])

        for word in bb_contents:
            if word in vocab:
                row.append(idx)
                col.append(vocab.get(word) - 1)
                data.append(1)
            else:
                if idx in alignments:
                    logging.info(f"Out of vocabulary: {word}")
                    stat[word] = stat.get(word, 0) + 1

    if embedding_type == "bow-general" and target == test:
        with open(f"./result/target-{target}-test-{test}-stat.json", "w") as fp:
            json.dump(stat, fp)

    return sp.coo_matrix((data, (row, col)), shape=(num_of_entities, len(vocab)))
