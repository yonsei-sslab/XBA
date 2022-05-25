program="libc"
epochs=200
python ./src/train.py --k 25 --layer 3 --target $program --embedding_type bow --seed 5 --epochs $epochs
python ./src/train.py --k 25 --layer 3 --target $program --embedding_type deepbindiff --seed 5 --epochs $epochs

program="libcrypto-xarch"
epochs=200
python ./src/train.py --k 25 --layer 3 --target $program --embedding_type bow --seed 5 --epochs $epochs
python ./src/train.py --k 25 --layer 3 --target $program --embedding_type deepbindiff --seed 5 --epochs $epochs

program="curl"
epochs=1000
python ./src/train.py --k 100 --layer 3 --target $program --embedding_type bow --seed 5 --epochs $epochs
python ./src/train.py --k 100 --layer 3 --target $program --embedding_type deepbindiff --seed 5 --epochs $epochs

program="httpd"
epochs=400
python ./src/train.py --k 100 --layer 3 --target $program --embedding_type bow --seed 5 --epochs $epochs
python ./src/train.py --k 100 --layer 3 --target $program --embedding_type deepbindiff --seed 5 --epochs $epochs

program="libcrypto"
epochs=1000
python ./src/train.py --k 50 --layer 3 --target $program --embedding_type bow --seed 5 --epochs $epochs
python ./src/train.py --k 50 --layer 3 --target $program --embedding_type deepbindiff --seed 5 --epochs $epochs

program="openssl"
epochs=100
python ./src/train.py --k 100 --layer 3 --target $program --embedding_type bow --seed 5 --epochs $epochs
python ./src/train.py --k 100 --layer 3 --target $program --embedding_type deepbindiff --seed 5 --epochs $epochs

program="sqlite3"
epochs=100
python ./src/train.py --k 100 --layer 3 --target $program --embedding_type bow --seed 5 --epochs $epochs
python ./src/train.py --k 100 --layer 3 --target $program --embedding_type deepbindiff --seed 5 --epochs $epochs


