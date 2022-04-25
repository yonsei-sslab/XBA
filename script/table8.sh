cd saved_model
rm general*
cd ..

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
                pipenv run -- python train_general.py --epochs 8000 --layer 3 --ae_dim 200 --target $target --test $test --embedding_type $type --seed 10 --norecord
            done
        fi
    done

    # Generate bow embedding for test data
    # Test on test data
    for type in bow-general
    do
        pipenv run -- python test_general.py --epochs 1000 --layer 3 --ae_dim 200 --target $test --test $test --embedding_type $type  --seed 0
    done
done