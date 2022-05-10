#!/bin/bash

program="libc"
epochs=200
python ./src/train.py --k 25 --layer 3 --target $program --embedding_type bow --seed 5 --epochs $epochs
python ./src/train.py --k 25 --layer 3 --target $program --embedding_type deepbindiff --seed 5 --epochs $epochs

program="libcrypto-xarch"
epochs=200
python ./src/train.py --k 25 --layer 3 --target $program --embedding_type bow --seed 5 --epochs $epochs
python ./src/train.py --k 25 --layer 3 --target $program --embedding_type deepbindiff --seed 5 --epochs $epochs
