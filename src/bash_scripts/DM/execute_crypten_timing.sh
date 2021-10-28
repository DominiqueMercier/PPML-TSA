#!/bin/bash
set -e -o pipefail

#./execute_crypten_timing.sh |& tee ../../../logs/execute_crypten_timing_output.txt

cd ../../scripts/DM/
echo "Current working dir: $PWD"

echo "=================================================="
echo "============= Start bash execution ==============="
echo "=================================================="

DATASETS=("ECG5000" "FordA" "ElectricDevices")

BATCHSIZES=(8)

for DATASET in ${DATASETS[@]}
do
    echo "=================================================="
    echo "Start experiments: $DATASET"
    echo "=================================================="

    for BATCHSIZE in ${BATCHSIZES[@]}
    do
        echo "=================================================="
        echo "Batch Size: $BATCHSIZE"
        echo "=================================================="

        python main_crypten_timing.py --verbose \
            --dataset_name $DATASET --standardize --validation_split 0.3 \
            --batch_size $BATCHSIZE --trys 10
    done
done

echo "=================================================="
echo "=========== Finished bash execution =============="
echo "=================================================="
