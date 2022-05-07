BASELINE_PROGRAM_LIST = openssl
BASELINE_PROPORTION_LIST = 5
BASELINE_EMBEDDING_LIST = deepbindiff

WINDOWS_PROGRAM_LIST = curl, 100, 8000, openssl, 100, 3000, sqlite3, 100, 7000, httpd, 100, 8000 libcrypto, 10, 15000
WINDOWS_PROPORTION_LIST = 5
WINDOWS_EMBEDDING_LIST = bow deepbindiff

ARM_PROGRAM_LIST = libc libcrypto-xarch
ARM_PROPORTION_LIST = 5
ARM_EMBEDDING_LIST = bow deepbindiff

table8:
	./script/table8.sh

table6:
	./script/table6-baseline-bow.sh
	./script/table6-baseline-dbd.sh
	./script/table6-baseline-innereye.sh
	./script/table6-x86-windows.sh
	./script/table6-x86-aarch64.sh

table7:
	python train.py --k 25 --layer 5 --target libcrypto-xarch --embedding_type bow --seed 10 --epochs 2000 --restore --norecord
	./script/table7.sh