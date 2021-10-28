#!/bin/bash
set -e -o pipefail

#./execute_federated_learning.sh |& tee ../../../logs/execute_federated_learning_adaptive_output.txt

cd ../../scripts/DM/
echo "Current working dir: $PWD"

echo "=================================================="
echo "============= Start bash execution ==============="
echo "=================================================="

DATASETS=('AsphaltPavementType' 'AsphaltRegularity' 'CharacterTrajectories' 'Crop' 'ECG5000' \
            'ElectricDevices' 'FaceDetection' 'FordA' 'HandOutlines' 'MedicalImages' 'MelbournePedestrian' \
            'NonInvasiveFetalECGThorax1' 'PhalangesOutlinesCorrect' 'Strawberry' 'UWaveGestureLibraryAll' 'Wafer')

MODELS=('AlexNet')

NCLIENTS=(2 4)
NPARALLEL=(2 4)
len=${#NCLIENTS[@]}

for DATASET in ${DATASETS[@]}
do
    for MODEL in ${MODELS[@]}
    do
        echo "=================================================="
        echo "Dataset: $DATASET | Model: $MODEL"
        echo "=================================================="

        for (( i=0; i<$len; i++ ))
        do 
            NCLIENT=${NCLIENTS[$i]}
            PARALLEL=${NPARALLEL[$i]}

            echo "=================================================="
            echo "NCLIENT: $NCLIENT PARALLEL: $PARALLEL"
            echo "=================================================="

            python main_federated_learning.py --verbose \
                --dataset_name $DATASET --exp_path 'federated_adaptive' --runs 5 --architecture $MODEL \
                --standardize --validation_split 0.3 \
                --save_model --epochs 100 --batch_size 8 --adaptive \
                --n_clients $NCLIENT --parallel_clients $PARALLEL --use_stratified \
                --save_report --save_mcr --load_model
        done
    done
done

echo "=================================================="
echo "=========== Finished bash execution =============="
echo "=================================================="
