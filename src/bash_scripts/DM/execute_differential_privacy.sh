#!/bin/bash
set -e -o pipefail

#./execute_differential_privacy.sh |& tee ../../../logs/execute_differential_privacy_output.txt

cd ../../scripts/DM/
echo "Current working dir: $PWD"

echo "=================================================="
echo "============= Start bash execution ==============="
echo "=================================================="

DATASETS=('AsphaltPavementType' 'AsphaltRegularity' 'CharacterTrajectories' 'Crop' 'ECG5000' \
            'ElectricDevices' 'FaceDetection' 'FordA' 'HandOutlines' 'MedicalImages' 'MelbournePedestrian' \
            'NonInvasiveFetalECGThorax1' 'PhalangesOutlinesCorrect' 'Strawberry' 'UWaveGestureLibraryAll' 'Wafer')

MODELS=('AlexNet')

L2_NORM_CLIPS=(0.5)
NOISE_MULTIPLIERS=(0.1)
len=${#L2_NORM_CLIPS[@]}

for DATASET in ${DATASETS[@]}
do
    for MODEL in ${MODELS[@]}
    do
        echo "=================================================="
        echo "Dataset: $DATASET | Model: $MODEL"
        echo "=================================================="

        for (( i=0; i<$len; i++ ))
        do 
            CLIP=${L2_NORM_CLIPS[$i]}
            NOISE=${NOISE_MULTIPLIERS[$i]}

            echo "=================================================="
            echo "L2_NORM_CLIP: $CLIP NOISE_MULTIPLIER: $NOISE"
            echo "=================================================="

            python main_differential_privacy.py --verbose \
                --dataset_name $DATASET --exp_path 'differential' --runs 5 --architecture $MODEL \
                --standardize --validation_split 0.3 \
                --save_model --epochs 100 --batch_size 16 \
                --l2_norm_clip $CLIP --noise_multiplier $NOISE \
                --save_report --save_mcr --load_model
        done
    done
done

echo "=================================================="
echo "=========== Finished bash execution =============="
echo "=================================================="
