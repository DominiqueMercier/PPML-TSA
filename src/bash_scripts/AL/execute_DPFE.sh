#!/bin/bash
set -e -o pipefail

#./execute_federated_learning.sh |& tee ../../../logs/execute_federated_learning_output.txt

cd ../../scripts/AL/
echo "Current working dir: $PWD"

echo "=================================================="
echo "============= Start bash execution ==============="
echo "=================================================="

DATASETS=('AsphaltPavementType' 'AsphaltRegularity' 'CharacterTrajectories' 'Crop' 'ECG5000' \
            'ElectricDevices' 'FaceDetection' 'FordA' 'HandOutlines' 'MedicalImages' 'MelbournePedestrian' \
            'NonInvasiveFetalECGThorax1' 'PhalangesOutlinesCorrect' 'Strawberry' 'UWaveGestureLibraryAll' 'Wafer')
NCLIENTS=(2 4)
BATCHSIZES=(8 16 32 64)

for DATASET in ${DATASETS[@]}
do
    for NCLIENT in ${NCLIENTS[@]}
    do 
        for BATCHSIZE in ${BATCHSIZES[@]}
        do
            echo "=============================================================="
            echo "Dataset: $DATASET; NCLIENT: $NCLIENT; Batchsize: $BATCHSIZE"
            echo "=============================================================="

            python3 main_DP-FE.py --verbose \
                --dataset_name $DATASET --architecture AlexNet --standardize --validation_split 0.3 \
                --save_model --epochs 100 --batch_size $BATCHSIZE \
                --n_clients $NCLIENT --use_stratified --learning_rate 0.2 \
                --save_report --save_mcr --runs 5 --load_model \
                --l2_norm_clip 0.5 --noise_multiplier 0.1
        done
    done
done

echo "=================================================="
echo "=========== Finished bash execution =============="
echo "=================================================="
