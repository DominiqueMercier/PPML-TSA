#!/bin/bash
set -e -o pipefail

#./execute_differential_privacy_detailed.sh |& tee ../../../logs/execute_differential_privacy_detailed_output.txt

cd ../../scripts/DM/
echo "Current working dir: $PWD"

echo "=================================================="
echo "============= Start bash execution ==============="
echo "=================================================="

DATASETS=("ECG5000" "ElectricDevices" "FordA")

L2_NORM_CLIPS=(1.0 1.0 1.0 1.0 1.0 1.0 1.0)
NOISE_MULTIPLIERS=(0.1 0.125 0.15 0.175 0.2 0.225 0.25)
len=${#L2_NORM_CLIPS[@]}

for DATASET in ${DATASETS[@]}
do
    echo "=================================================="
    echo "Start experiments: $DATASET"
    echo "=================================================="

    for (( i=0; i<$len; i++ ))
    do 
        CLIP=${L2_NORM_CLIPS[$i]}
        NOISE=${NOISE_MULTIPLIERS[$i]}

        echo "=================================================="
        echo "L2_NORM_CLIP: $CLIP NOISE_MULTIPLIER: $NOISE"
        echo "=================================================="

        python main_differential_privacy.py --verbose \
            --dataset_name $DATASET --standardize --validation_split 0.3 \
            --save_model --epochs 100 --batch_size 32 \
            --l2_norm_clip $CLIP --noise_multiplier $NOISE \
            --save_report --load_model
    done
done

echo "=================================================="
echo "=========== Finished bash execution =============="
echo "=================================================="
