#!/bin/bash
PROGRAM_LIST="curl openssl sqlite3 httpd  libcrypto"
EMBEDDING_LIST="bow deepbindiff"

epochs=10
k=1

program="curl"

python train.py --k $k --layer 5 --target $program --embedding_type bow --seed 5 --epochs $epochs --norecord
python train.py --k $k --layer 5 --target $program --embedding_type deepbindiff --seed 5 --epochs $epochs --norecord

program="openssl"

python train.py --k $k --layer 5 --target $program --embedding_type bow --seed 5 --epochs $epochs --norecord
python train.py --k $k --layer 5 --target $program --embedding_type deepbindiff --seed 5 --epochs $epochs --norecord

program="sqlite3"

python train.py --k $k --layer 5 --target $program --embedding_type bow --seed 5 --epochs $epochs --norecord
python train.py --k $k --layer 5 --target $program --embedding_type deepbindiff --seed 5 --epochs $epochs --norecord

program="httpd"

python train.py --k $k --layer 5 --target $program --embedding_type bow --seed 5 --epochs $epochs --norecord
python train.py --k $k --layer 5 --target $program --embedding_type deepbindiff --seed 5 --epochs $epochs --norecord

program="libcrypto"

python train.py --k $k --layer 5 --target $program --embedding_type bow --seed 5 --epochs $epochs --norecord
python train.py --k $k --layer 5 --target $program --embedding_type deepbindiff --seed 5 --epochs $epochs --norecord

program="libcrypto-xarch"
python train.py --k $k --layer 5 --target $program --embedding_type bow --seed 5 --epochs $epochs --norecord
python train.py --k $k --layer 5 --target $program --embedding_type deepbindiff --seed 5 --epochs $epochs --norecord

program="libc"
python train.py --k $k --layer 5 --target $program --embedding_type bow --seed 5 --epochs $epochs --norecord
python train.py --k $k --layer 5 --target $program --embedding_type deepbindiff --seed 5 --epochs $epochs --norecord
