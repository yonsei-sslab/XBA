#!/bin/bash
PROGRAM_LIST="curl openssl sqlite3 httpd  libcrypto"
EMBEDDING_LIST="bow deepbindiff"

program="curl"
epochs=1000
python train.py --k 100 --layer 5 --target $program --embedding_type bow --seed 5 --epochs $epochs
python train.py --k 100 --layer 5 --target $program --embedding_type deepbindiff --seed 5 --epochs $epochs

program="openssl"
epochs=1000
python train.py --k 100 --layer 5 --target $program --embedding_type bow --seed 5 --epochs $epochs
python train.py --k 100 --layer 5 --target $program --embedding_type deepbindiff --seed 5 --epochs $epochs

program="sqlite3"
epochs=1000
python train.py --k 100 --layer 5 --target $program --embedding_type bow --seed 5 --epochs $epochs
python train.py --k 100 --layer 5 --target $program --embedding_type deepbindiff --seed 5 --epochs $epochs

program="httpd"
epochs=1000
python train.py --k 100 --layer 5 --target $program --embedding_type bow --seed 5 --epochs $epochs
python train.py --k 100 --layer 5 --target $program --embedding_type deepbindiff --seed 5 --epochs $epochs
comment

program="libcrypto"
epochs=1000
python train.py --k 50 --layer 5 --target $program --embedding_type bow --seed 5 --epochs $epochs
python train.py --k 50 --layer 5 --target $program --embedding_type deepbindiff --seed 5 --epochs $epochs

