#!/bin/bash
set -e -o pipefail

#./execute_federated_learning.sh |& tee ../../../logs/execute_federated_learning_output.txt

cd ../../scripts/AL/
echo "Current working dir: $PWD"

echo "=================================================="
echo "============= Start bash execution ==============="
echo "=================================================="

DATASETS=('ECG5000' 'ElectricDevices' 'FordA')
MODELS=('AlexNet' 'LeNet' 'FCN' 'FDN' 'LSTM')

for DATASET in ${DATASETS[@]}
do
    for MODEL in ${MODELS[@]}
    do
        echo "=================================================="
        echo "Dataset: $DATASET; Model: $MODEL"
        echo "=================================================="

        python3 main_federated_ensemble.py --verbose \
            --dataset_name $DATASET --architecture $MODEL --standardize --validation_split 0.3 \
            --save_model --epochs 100 --batch_size 8 \
            --n_clients 4 --use_stratified --initial_lr 0.01 \
            --save_report --save_mcr --runs 1
    done
done

echo "=================================================="
echo "=========== Finished bash execution =============="
echo "=================================================="
