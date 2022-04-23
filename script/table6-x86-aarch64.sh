#!/bin/bash
PROGRAM_LIST="libc libcrypto-xarch"
EMBEDDING_LIST="bow deepbindiff"

program="libcrypto-xarch"
epochs=10000
python train.py --k 50 --layer 5 --target $program --embedding_type bow --seed 5 --epochs $epochs
python train.py --k 50 --layer 5 --target $program --embedding_type deepbindiff --seed 5 --epochs $epochs

program="libc"
epochs=15000
python train.py --k 100 --layer 5 --target $program --embedding_type bow --seed 5 --epochs $epochs
python train.py --k 100 --layer 5 --target $program --embedding_type deepbindiff --seed 5 --epochs $epochs

