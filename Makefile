table8:
	./script/table8.sh

table6:
	./script/table6-baseline-bow.sh
	./script/table6-baseline-dbd.sh
	./script/table6-baseline-innereye.sh
	./script/table6-x86-windows.sh
	./script/table6-x86-aarch64.sh

table7:
	python ./src/train.py --k 25 --layer 5 --target libcrypto-xarch --embedding_type bow --seed 10 --epochs 200 --norestore --norecord --novalidate
	python ./src/train.py --k 25 --layer 5 --target libcrypto --embedding_type bow --seed 10 --epochs 50 --norestore --norecord --novalidate
	python ./src/train.py --k 50 --layer 5 --target openssl --embedding_type bow --seed 10 --epochs 10 --norestore --norecord --novalidate
	python ./src/train.py --k 50 --layer 5 --target sqlite3 --embedding_type bow --seed 10 --epochs 130 --norestore --norecord --novalidate
	./script/table7.sh

figure3:
	./script/figure3.sh

test:
	./script/test-run.sh


