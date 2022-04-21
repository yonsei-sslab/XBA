from utils import get_git_root
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
    data_root_dir_path = get_git_root("utils.py") + "/data/done/" + program
    disasm_path = data_root_dir_path + "/disasm_innereye.json"
    f = open(disasm_path, "r")
    bb_list = json.load(f)

    vocab = {}
    counter.value = 0
    vocab[""] = counter()
    for idx, bb in bb_list.items():
        for instruction in bb:
            if instruction not in vocab:
                vocab[instruction] = counter()

    with open(f"../vocabulary/{program}_vocabulary_for_new_alignments.json", "w") as fp:
        json.dump(vocab, fp)
