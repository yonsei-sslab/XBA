#!/bin/bash
PROGRAM_LIST="libc libcrypto-xarch"
EMBEDDING_LIST="bow deepbindiff"

program="libc"
epochs=15000
python train.py --k 100 --layer 3 --target $program --embedding_type bow --seed 5 --epochs $epochs
python train.py --k 100 --layer 3 --target $program --embedding_type deepbindiff --seed 5 --epochs $epochs

program="libcrypto-xarch"
epochs=15000
python train.py --k 50 --layer 3 --target $program --embedding_type bow --seed 5 --epochs $epochs
python train.py --k 50 --layer 3 --target $program --embedding_type deepbindiff --seed 5 --epochs $epochs
