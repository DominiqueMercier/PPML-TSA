#!/bin/bash
set -e -o pipefail

#./execute_crypten.sh |& tee ../../../logs/execute_crypten_output.txt

cd ../../scripts/DM/
echo "Current working dir: $PWD"

echo "=================================================="
echo "============= Start bash execution ==============="
echo "=================================================="

DATASETS=("ECG5000")

NCLIENTS=(1)
BATCHSIZES=(8)
len=${#NCLIENTS[@]}

for DATASET in ${DATASETS[@]}
do
    echo "=================================================="
    echo "Start experiments: $DATASET"
    echo "=================================================="

    for (( i=0; i<$len; i++ ))
    do 
        NCLIENT=${NCLIENTS[$i]}

        for BATCHSIZE in ${BATCHSIZES[@]}
        do
            echo "=================================================="
            echo "NCLIENT: $NCLIENT | Batch Size: $BATCHSIZE"
            echo "=================================================="

            python main_crypten.py --verbose \
                --dataset_name $DATASET --standardize --validation_split 0.3 \
                --save_model --epochs 1 --batch_size $BATCHSIZE \
                --n_clients $NCLIENT
        done
    done
done

echo "=================================================="
echo "=========== Finished bash execution =============="
echo "=================================================="
