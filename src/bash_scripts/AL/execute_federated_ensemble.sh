#!/bin/bash
set -e -o pipefail

#./execute_federated_learning.sh |& tee ../../../logs/execute_federated_learning_output.txt
ls
cd ../../scripts/AL/
echo "Current working dir: $PWD"

echo "=================================================="
echo "============= Start bash execution ==============="
echo "=================================================="

DATASETS=('AsphaltPavementType' 'AsphaltRegularity' 'CharacterTrajectories' 'Crop' 'ECG5000' \
            'ElectricDevices' 'FaceDetection' 'FordA' 'HandOutlines' 'MedicalImages' 'MelbournePedestrian' \
            'NonInvasiveFetalECGThorax1' 'PhalangesOutlinesCorrect' 'Strawberry' 'UWaveGestureLibraryAll' 'Wafer')
NCLIENTS=(2 4)

for DATASET in ${DATASETS[@]}
do
    for NCLIENT in ${NCLIENTS[@]}
    do
        echo "=================================================="
        echo "Dataset: $DATASET; NCLIENT: $NCLIENT"
        echo "=================================================="

        python3 main_federated_ensemble.py --verbose \
            --dataset_name $DATASET --standardize --validation_split 0.3 \
            --save_model --epochs 100 --batch_size 8 \
            --n_clients $NCLIENT --use_stratified --initial_lr 0.01 \
            --save_report --save_mcr --runs 5 --architecture AlexNet
    done
done

echo "=================================================="
echo "=========== Finished bash execution =============="
echo "=================================================="
