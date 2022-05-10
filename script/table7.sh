# sqlite3
python ./src/get_rank.py --seed 10 --layer 5 --k 50 --ae_dim 200 --learning_rate 0.001 --gamma 3 --epochs 130 --target sqlite3 --embedding_type bow --bb_id1="269, 270, 271" --bb_id2="40621"
python ./src/get_rank.py --seed 10 --layer 5 --k 50 --ae_dim 200 --learning_rate 0.001 --gamma 3 --epochs 130 --target sqlite3 --embedding_type bow --bb_id1="106, 107, 108" --bb_id2="49939, 78515"
python ./src/get_rank.py --seed 10 --layer 5 --k 50 --ae_dim 200 --learning_rate 0.001 --gamma 3 --epochs 130 --target sqlite3 --embedding_type bow --bb_id1="67, 68" --bb_id2="44888"
python ./src/get_rank.py --seed 10 --layer 5 --k 50 --ae_dim 200 --learning_rate 0.001 --gamma 3 --epochs 130 --target sqlite3 --embedding_type bow --bb_id1="10" --bb_id2="46380"
python ./src/get_rank.py --seed 10 --layer 5 --k 50 --ae_dim 200 --learning_rate 0.001 --gamma 3 --epochs 130 --target sqlite3 --embedding_type bow --bb_id1="80109" --bb_id2="80110"

# libcrypto
python ./src/get_rank.py --seed 10 --layer 5 --k 25 --ae_dim 200 --learning_rate 0.001 --gamma 3 --epochs 50 --target libcrypto --embedding_type bow --bb_id1="146, 147, 148" --bb_id2="162079"
python ./src/get_rank.py --seed 10 --layer 5 --k 25 --ae_dim 200 --learning_rate 0.001 --gamma 3 --epochs 50 --target libcrypto --embedding_type bow --bb_id1="266, 267, 268" --bb_id2="162079"
python ./src/get_rank.py --seed 10 --layer 5 --k 25 --ae_dim 200 --learning_rate 0.001 --gamma 3 --epochs 50 --target libcrypto --embedding_type bow --bb_id1="401, 402, 403" --bb_id2="163219"
python ./src/get_rank.py --seed 10 --layer 5 --k 25 --ae_dim 200 --learning_rate 0.001 --gamma 3 --epochs 50 --target libcrypto --embedding_type bow --bb_id1="14, 15" --bb_id2="163222"
python ./src/get_rank.py --seed 10 --layer 5 --k 25 --ae_dim 200 --learning_rate 0.001 --gamma 3 --epochs 50 --target libcrypto --embedding_type bow --bb_id1="209, 210" --bb_id2="163227"
python ./src/get_rank.py --seed 10 --layer 5 --k 25 --ae_dim 200 --learning_rate 0.001 --gamma 3 --epochs 50 --target libcrypto --embedding_type bow --bb_id1="311, 312" --bb_id2="163236"


# openssl
python ./src/get_rank.py --seed 10 --layer 5 --k 50 --ae_dim 200 --learning_rate 0.001 --gamma 3 --epochs 10 --target openssl --embedding_type bow --bb_id1="1454, 1455, 1456, 8406, 20381" --bb_id2="35147"
python ./src/get_rank.py --seed 10 --layer 5 --k 50 --ae_dim 200 --learning_rate 0.001 --gamma 3 --epochs 10 --target openssl --embedding_type bow --bb_id1="809, 810" --bb_id2="35155" # read
python ./src/get_rank.py --seed 10 --layer 5 --k 50 --ae_dim 200 --learning_rate 0.001 --gamma 3 --epochs 10 --target openssl --embedding_type bow --bb_id1="3608, 3609" --bb_id2="35160"
python ./src/get_rank.py --seed 10 --layer 5 --k 50 --ae_dim 200 --learning_rate 0.001 --gamma 3 --epochs 10 --target openssl --embedding_type bow --bb_id1="118, 119" --bb_id2="35154" # fileno
python ./src/get_rank.py --seed 10 --layer 5 --k 50 --ae_dim 200 --learning_rate 0.001 --gamma 3 --epochs 10 --target openssl --embedding_type bow --bb_id1="3794, 3795" --bb_id2="35094"
python ./src/get_rank.py --seed 10 --layer 5 --k 50 --ae_dim 200 --learning_rate 0.001 --gamma 3 --epochs 10 --target openssl --embedding_type bow --bb_id1="644, 645, 646" --bb_id2="21669" # strcasecmp


# libcrypto-xarch
python ./src/get_rank.py --layer 5 --k 25 --seed 10 --learning_rate 0.001 --gamma 3 --epochs 200 --ae_dim 200 --target libcrypto-xarch --embedding_type bow --bb_id1="93448, 93449" --bb_id2="197104, 197105" # aesni_gcm_initkey
python ./src/get_rank.py --layer 5 --k 25 --seed 10 --learning_rate 0.001 --gamma 3 --epochs 200 --ae_dim 200 --target libcrypto-xarch --embedding_type bow --bb_id1="62957, 62959, 62958, 62960, 62682" --bb_id2="168486, 168731" # gcm_init_avx
python ./src/get_rank.py --layer 5 --k 25 --seed 10 --learning_rate 0.001 --gamma 3 --epochs 200 --ae_dim 200 --target libcrypto-xarch --embedding_type bow --bb_id1="74744, 74748, 74749, 74735" --bb_id2="179814, 179815, 179816, 179809 " # sha1_block_data_order_shaext
python ./src/get_rank.py --layer 5 --k 25 --seed 10 --learning_rate 0.001 --gamma 3 --epochs 200 --ae_dim 200 --target libcrypto-xarch --embedding_type bow --bb_id1="74900, 74904, 74905, 74888" --bb_id2="179910, 179912, 179913, 179901 " # sha256_block_data_order_shaext
python ./src/get_rank.py --layer 5 --k 25 --seed 10 --learning_rate 0.001 --gamma 3 --epochs 200 --ae_dim 200 --target libcrypto-xarch --embedding_type bow --bb_id1="36366, 36368, 36369, 36370, 36373" --bb_id2="143466, 143467, 143472, 143468, 143471" # pointadd 
python ./src/get_rank.py --layer 5 --k 25 --seed 10 --learning_rate 0.001 --gamma 3 --epochs 200 --ae_dim 200 --target libcrypto-xarch --embedding_type bow --bb_id1="62271, 62272, 62274, 62276, 62275, 62273, 45274 " --bb_id2="197090, 197091, 197093, 197095, 197092,197094, 197096, 151798" # aesni_gcm_encrypt
python ./src/get_rank.py --layer 5 --k 25 --seed 10 --learning_rate 0.001 --gamma 3 --epochs 200 --ae_dim 200 --target libcrypto-xarch --embedding_type bow --bb_id1="62259, 62260, 62262, 62264, 62263, 62261, 45289" --bb_id2="197097, 197098, 197100, 197102, 197099, 197101, 197103, 151810"  # aesni_gcm_decrypt
python ./src/get_rank.py --layer 5 --k 25 --seed 10 --learning_rate 0.001 --gamma 3 --epochs 200 --ae_dim 200 --target libcrypto-xarch --embedding_type bow --bb_id1="17435, 17436, 17437, 17438, 17439, 17440, 17441, 17442, 17443, 17444" --bb_id2="125554, 125555, 125556, 125557, 125558, 125559, 125560, 125561, 125562, 125563" # ekeygen
python ./src/get_rank.py --layer 5 --k 25 --seed 10 --learning_rate 0.001 --gamma 3 --epochs 200 --ae_dim 200 --target libcrypto-xarch --embedding_type bow --bb_id1="1161, 1162, 1164, 1165, 1167, 1169, 1171, 1172, 1175, 1176, 1168, 1177, 1178, 1182, 1183, 1166, 1184, 1185, 1189, 1190, 1191, 1170, 1163, 1155" --bb_id2="109329, 109330, 109332, 109333, 109334, 109335, 109336, 109338, 109341, 109337, 109342, 109344, 109339, 109345, 109346, 109347, 109340, 109348, 109349, 109343, 109331, 109350" # encryptkey
python ./src/get_rank.py --layer 5 --k 25 --seed 10 --learning_rate 0.001 --gamma 3 --epochs 200 --ae_dim 200 --target libcrypto-xarch --embedding_type bow --bb_id1="986, 987, 988, 989, 991, 992, 993, 994, 990, 995, 997, 999, 1001, 1003, 998, 1005, 1006, 200, 1002, 1004, 996, 1007, 1009, 1010, 1011, 1012, 1008, 1013" --bb_id2="109472, 109473, 109475, 109476, 109477, 109478, 109480, 109474, 109481, 109483, 109484, 109485, 109487, 109490, 109489, 109491, 109492, 109494, 109495, 109496, 109497, 109499, 109500, 109493, 109501, 109488, 109502, 109503, 109486, 109504, 109505, 109506, 109507, 109508, 109509, 109498, 109510, 109511, 109512, 109513, 109514, 109482, 109479, 109515 " # aesni_xts_encrypt


