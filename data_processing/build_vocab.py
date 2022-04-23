import utils
import json
import pandas as pd

programs = [
    "curl",
    "openssl",
    "httpd",
    "sqlite3",
    "libcrypto",
    "libc",
    "libcrypto-xarch",
]


def counter():
    current_num = counter.value
    counter.value += 1
    return current_num


# Generate individual vocabulary
for program in programs:
    data_root_dir_path = utils.get_git_root("utils.py") + "/data/done/" + program
    disasm_path = data_root_dir_path + "/disasm_innereye.json"
    seed_path = data_root_dir_path + "/alignment.csv"
    alignments = pd.read_csv(seed_path, header=None)
    f = open(disasm_path, "r")
    bb_list = json.load(f)

    vocab = {}
    counter.value = 0
    vocab[""] = counter()
    for idx, row in alignments.iterrows():
        for bb_id in row:
            bb = bb_list[str(bb_id)]
            for instruction in bb:
                if instruction not in vocab and len(bb) != 1:
                    vocab[instruction] = counter()

    with open(f"./vocabulary/{program}_vocabulary.json", "w") as fp:
        json.dump(vocab, fp)


programs = [
    "curl",
    "openssl",
    "httpd",
    "sqlite3",
    "libcrypto",
]

# Generate combined vocabulary for general AE model
for test in programs:
    vocab = {}
    counter.value = 0
    vocab[""] = counter()

    for target in programs:
        if target == test:
            continue

        data_root_dir_path = utils.get_git_root("utils.py") + "/data/done/" + target
        disasm_path = data_root_dir_path + "/disasm_innereye.json"
        seed_path = data_root_dir_path + "/alignment.csv"
        alignments = pd.read_csv(seed_path, header=None)
        f = open(disasm_path, "r")
        bb_list = json.load(f)

        for idx, row in alignments.iterrows():
            for bb_id in row:
                bb = bb_list[str(bb_id)]
                for instruction in bb:
                    if instruction not in vocab and len(bb) != 1:
                        vocab[instruction] = counter()

    with open(
        f'./vocabulary/{"-".join([x for x in programs if x != test])}_vocabulary.json',
        "w",
    ) as fp:
        json.dump(vocab, fp)
