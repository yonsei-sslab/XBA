#!/bin/bash
SEED="1 2 3 4"
program="libcrypto"
epochs=1000

for seed in $SEED; do 
	python ./src/train.py --k 50 --layer 3 --target $program --embedding_type bow --seed $seed --epochs $epochs 
done

program="libcrypto-xarch"
epochs=200

for seed in $SEED; do 
	python ./src/train.py --k 25 --layer 3 --target $program --embedding_type bow --seed $seed --epochs $epochs
done


