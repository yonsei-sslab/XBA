#!/bin/bash

program="libcrypto"
epochs=1000
python train.py --k 50 --layer 3 --target $program --embedding_type bow --seed 5 --epochs $epochs
python train.py --k 50 --layer 3 --target $program --embedding_type deepbindiff --seed 5 --epochs $epochs

program="curl"
epochs=100
python train.py --k 100 --layer 3 --target $program --embedding_type bow --seed 5 --epochs $epochs
python train.py --k 100 --layer 3 --target $program --embedding_type deepbindiff --seed 5 --epochs $epochs

program="openssl"
epochs=100
python train.py --k 100 --layer 3 --target $program --embedding_type bow --seed 5 --epochs $epochs
python train.py --k 100 --layer 3 --target $program --embedding_type deepbindiff --seed 5 --epochs $epochs

program="sqlite3"
epochs=100
python train.py --k 100 --layer 3 --target $program --embedding_type bow --seed 5 --epochs $epochs
python train.py --k 100 --layer 3 --target $program --embedding_type deepbindiff --seed 5 --epochs $epochs

program="httpd"
epochs=100
python train.py --k 100 --layer 3 --target $program --embedding_type bow --seed 5 --epochs $epochs
python train.py --k 100 --layer 3 --target $program --embedding_type deepbindiff --seed 5 --epochs $epochs



