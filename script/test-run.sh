#!/bin/bash

epochs=10
k=1

program="curl"

python ./src/train.py --k $k --layer 5 --target $program --embedding_type bow --seed 5 --epochs $epochs --norecord
python ./src/train.py --k $k --layer 5 --target $program --embedding_type deepbindiff --seed 5 --epochs $epochs --norecord

program="openssl"

python ./src/train.py --k $k --layer 5 --target $program --embedding_type bow --seed 5 --epochs $epochs --norecord
python ./src/train.py --k $k --layer 5 --target $program --embedding_type deepbindiff --seed 5 --epochs $epochs --norecord

program="sqlite3"

python ./src/train.py --k $k --layer 5 --target $program --embedding_type bow --seed 5 --epochs $epochs --norecord
python ./src/train.py --k $k --layer 5 --target $program --embedding_type deepbindiff --seed 5 --epochs $epochs --norecord

program="httpd"

python ./src/train.py --k $k --layer 5 --target $program --embedding_type bow --seed 5 --epochs $epochs --norecord
python ./src/train.py --k $k --layer 5 --target $program --embedding_type deepbindiff --seed 5 --epochs $epochs --norecord

program="libcrypto"

python ./src/train.py --k $k --layer 5 --target $program --embedding_type bow --seed 5 --epochs $epochs --norecord
python ./src/train.py --k $k --layer 5 --target $program --embedding_type deepbindiff --seed 5 --epochs $epochs --norecord

program="libcrypto-xarch"
python ./src/train.py --k $k --layer 5 --target $program --embedding_type bow --seed 5 --epochs $epochs --norecord
python ./src/train.py --k $k --layer 5 --target $program --embedding_type deepbindiff --seed 5 --epochs $epochs --norecord

program="libc"
python ./src/train.py --k $k --layer 5 --target $program --embedding_type bow --seed 5 --epochs $epochs --norecord
python ./src/train.py --k $k --layer 5 --target $program --embedding_type deepbindiff --seed 5 --epochs $epochs --norecord
