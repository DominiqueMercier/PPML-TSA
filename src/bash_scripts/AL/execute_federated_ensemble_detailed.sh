#!/bin/bash
source ~/venv/PPML/bin/activate

set -e -o pipefail

#./execute_federated_learning.sh |& tee ../../../logs/execute_federated_learning_output.txt

cd ../../scripts/AL/
echo "Current working dir: $PWD"

echo "=================================================="
echo "============= Start bash execution ==============="
echo "=================================================="

DATASETS=("ECG5000" "FordA" "ElectricDevices")
NCLIENTS=(2 3 4 5 6 7 8 9 10)
BATCHSIZES=(8 16 32 64)
LRS=(0.01 0.001 0.0001)

for DATASET in ${DATASETS[@]}
do
    for NCLIENT in ${NCLIENTS[@]}
    do 
        for LR in ${LRS[@]}
        do
            for BATCHSIZE in ${BATCHSIZES[@]}
            do
                echo "=================================================="
                echo "Dataset: $DATASET; NCLIENT: $NCLIENT; LR: $LR; Batchsize: $BATCHSIZE"
                echo "=================================================="

                python3 main_federated_ensemble.py --verbose \
                    --dataset_name $DATASET --standardize --validation_split 0.3 \
                    --save_model --epochs 100 --batch_size $BATCHSIZE \
                    --n_clients $NCLIENT --use_stratified --initial_lr $LR \
                    --save_report --runs 1
            done
        done
    done
done

echo "=================================================="
echo "=========== Finished bash execution =============="
echo "=================================================="
