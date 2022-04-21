#!/bin/bash
PROGRAM_LIST="curl openssl sqlite3 httpd  libcrypto libc libcrypto-xarch"

for program in $PROGRAM_LIST; do 
	python baseline.py --target $program --embedding_type innereye; \
done