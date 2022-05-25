for test in sqlite3
do
    # Generate bow embedding for targets data
    # Train the general model for AE on targets data

    for target in curl sqlite3 httpd libcrypto openssl
    do
        if [ "$test" != "$target" ]
        then
            for type in bow-general
            do
                python ./src/train.py --model_name general-model --epochs 200 --layer 3 --ae_dim 200 --target $target --embedding_type $type --test_target $test --seed 10 --k 25 --norecord --novalidate --restore
            done
        fi
    done

    # Generate bow embedding for test data
    # Test on test data
    for type in bow-general
    do
        python ./src/test.py --model_name general-model --epochs 200 --layer 3 --ae_dim 200 --target $test --test_target $test --embedding_type $type  --seed 0
    done
done