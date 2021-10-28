#!/bin/bash
set -e -o pipefail

#./execute_federated_learning_adaptive_sgd_detailed.sh |& tee ../../../logs/execute_federated_learning_adaptive_sgd_detailed_output.txt

cd ../../scripts/DM/
echo "Current working dir: $PWD"

echo "=================================================="
echo "============= Start bash execution ==============="
echo "=================================================="

DATASETS=("ECG5000" "ElectricDevices" "FordA")

NCLIENTS=(4 4 4)
NPARALLEL=(2 3 4)
len=${#NCLIENTS[@]}

for DATASET in ${DATASETS[@]}
do
    echo "=================================================="
    echo "Start experiments: $DATASET"
    echo "=================================================="

    for (( i=0; i<$len; i++ ))
    do 
        NCLIENT=${NCLIENTS[$i]}
        PARALLEL=${NPARALLEL[$i]}

        echo "=================================================="
        echo "NCLIENT: $NCLIENT PARALLEL: $PARALLEL"
        echo "=================================================="

        python main_federated_learning.py --verbose \
            --dataset_name $DATASET --standardize --validation_split 0.3 \
            --save_model --epochs 1000 --batch_size 32 \
             --stepwise --adaptive \
            --n_clients $NCLIENT --parallel_clients $PARALLEL --use_stratified \
            --save_report --load_model
        
        python main_federated_learning.py --verbose \
            --dataset_name $DATASET --standardize --validation_split 0.3 \
            --save_model --epochs 1000 --batch_size 32 \
            --stepwise --adaptive \
            --n_clients $NCLIENT --parallel_clients $PARALLEL \
            --save_report --load_model
    done
done

echo "=================================================="
echo "=========== Finished bash execution =============="
echo "=================================================="
