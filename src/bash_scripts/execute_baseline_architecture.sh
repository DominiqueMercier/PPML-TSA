#!/bin/bash
set -e -o pipefail

#./execute_baseline_architecture.sh |& tee ../../logs/execute_baseline_architecture_output.txt

cd ../scripts/
echo "Current working dir: $PWD"

echo "=================================================="
echo "============= Start bash execution ==============="
echo "=================================================="

DATASETS=('ECG5000' 'ElectricDevices' 'FordA')

MODELS=('LeNet' 'FCN' 'FDN' 'LSTM')

for DATASET in ${DATASETS[@]}
do
    for MODEL in ${MODELS[@]}
    do
        echo "=================================================="
        echo "Dataset: $DATASET | Model: $MODEL"
        echo "=================================================="

        python main_baseline.py --verbose \
            --dataset_name $DATASET --exp_path 'baseline_architecture' --runs 1 --architecture $MODEL \
            --standardize --validation_split 0.3 \
            --save_model --epochs 100 --batch_size 8 \
            --save_report --save_mcr --load_model
    done
done

echo "=================================================="
echo "=========== Finished bash execution =============="
echo "=================================================="
