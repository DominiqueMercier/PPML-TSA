import os
import sys
import random
from optparse import OptionParser

import numpy as np
import tensorflow as tf
from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasSGDOptimizer

############ Tensforflow config ############
tf.compat.v1.disable_eager_execution()

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)

############## Import modules ##############
sys.path.append("../../")
from modules.model_definition import FCN, FDN, LSTM, AlexNet, LeNet
from modules.DM import trainer_differential_privacy
from modules import ucr_loader, utils, mean_cr_utils
from modules.AL.utils import make_federated_data, make_directory_if_not_exists
from modules.AL.keras_utils import test_keras_ensemble

def process(options):

    # Loop over N runs
    report_paths = []
    for i in range(options.runs):
        ######### Global Run Settings ###########
        np.random.seed(i)
        tf.random.set_seed(i)
        random.seed(i)

        if options.verbose:
            print('Run %d / %d' % (i+1, options.runs))

        model_dir, result_dir = os.path.join('../../../', 'models'), os.path.join('../../../', 'results')

        ######### Dataset processing ###########
        # get all datasets
        dataset_dict = ucr_loader.get_datasets(options.root_path, prefix='**/')
        # retrieve data
        dataset_name = options.dataset_name
        trainX, trainY, testX, testY = ucr_loader.load_data(dataset_dict[dataset_name])
        # preprocess data
        trainX, trainY, testX, testY = ucr_loader.preprocess_data(trainX, trainY, testX, testY, normalize=options.normalize, standardize=options.standardize)
        # additional preprocessing
        trainX, trainY, valX, valY = utils.perform_datasplit(trainX, trainY, test_split=options.validation_split)
        n_classes = len(np.unique(trainY))

        if options.verbose:
            print('TrainX:', trainX.shape)
            if options.validation_split > 0:
                print('ValX:', valX.shape)
            print('TestX:', testX.shape)
            print('Classes:', n_classes)

        # Split data
        federated_trainX, federated_trainY = make_federated_data(trainX, trainY, n_clients=options.n_clients, stratify=options.use_stratified)
        federated_valX, federated_valY = make_federated_data(valX, valY, n_clients=options.n_clients, stratify=options.use_stratified)

        # Prepare test data
        num_test_samples = testX.shape[0]
        testdata_shape = testX.shape[1:]
        test_size = int(num_test_samples/options.batch_size) * options.batch_size
        if test_size < num_test_samples:
            fill = np.zeros((options.batch_size - (num_test_samples - test_size), *testdata_shape))
            testX = np.concatenate([testX, fill], axis=0)

        lst_model_paths = []

        for CLIENT_ID in range(options.n_clients):
            
            # Get corresponding data fold
            trainX = federated_trainX[CLIENT_ID]
            trainY = federated_trainY[CLIENT_ID]
            valX = federated_valX[CLIENT_ID]
            valY = federated_valY[CLIENT_ID]

            print('TrainX:', trainX.shape)

            sequence_length = trainX.shape[-1]
            sequence_channels = trainX.shape[1]
            sequence_shape = trainX.shape[1:]

            #### Prepare differential datasets #####
            train_size, val_size = int(trainX.shape[0]/options.batch_size) * options.batch_size, int(valX.shape[0]/options.batch_size) * options.batch_size
            val_size = options.batch_size if val_size < options.batch_size else val_size
            trainX, trainY, valX, valY = trainX[:train_size], trainY[:train_size], valX[:val_size], valY[:val_size]

            experiment_dirname = 'DPFE_%s/%s/BS-%d_LR-%.4f_l2-%.4f_noise-%.4f_NClients-%d/'%(dataset_name, options.architecture, options.batch_size, options.learning_rate, options.l2_norm_clip, options.noise_multiplier, options.n_clients)
            
            result_run_dir = os.path.join(result_dir, experiment_dirname, 'run-%d'%i)
            make_directory_if_not_exists(result_run_dir)

            model_run_dir = os.path.join(model_dir, experiment_dirname, 'run-%d'%i)
            make_directory_if_not_exists(model_run_dir)

            ###### Train differential model ########
            model_path = os.path.join(model_run_dir, 'Client-%d'%CLIENT_ID + '.h5') if options.save_model else None
            lst_model_paths.append(model_path)

            ##### model architecture ######
            architecture_func = {'AlexNet': AlexNet().build_1d, 'LSTM': LSTM().build_default,
                                    'FCN': FCN().build_default, 'FDN': FDN().build_default, 'LeNet': LeNet().build_default}

            if os.path.exists(model_path) and options.load_model:
                model = architecture_func[options.architecture](sequence_shape, n_classes, activation='softmax', verbose=1)
                model.load_weights(model_path)
            else:
                model = architecture_func[options.architecture](sequence_shape, n_classes, activation='softmax', verbose=1)
                optimizer = DPKerasSGDOptimizer(l2_norm_clip=options.l2_norm_clip, noise_multiplier=options.noise_multiplier, num_microbatches=options.batch_size, learning_rate=options.learning_rate)
                loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.losses.Reduction.NONE)
                model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
                trainer_differential_privacy.traintf(sess, model, trainX, trainY, validation_data=(valX, valY), epochs=options.epochs, batch_size=options.batch_size, model_path=model_path, verbose=options.verbose)

            ############# Evaluation ###############
            report_path = os.path.join(result_run_dir, 'Client-%d'%CLIENT_ID + '_report.txt') if options.save_report else None

            preds = np.argmax(model.predict(testX, batch_size=options.batch_size,
                                            verbose=options.verbose), axis=1)
            del model
            
            utils.compute_classification_report(
                testY, preds[:num_test_samples], save=report_path, verbose=options.verbose, store_dict=False)

            del trainX, trainY, valX, valY

        del federated_trainX, federated_trainY
        del federated_valX, federated_valY

        # Load all models for ensembling
        models = []
        for model_path in lst_model_paths:
            tmp_model = architecture_func[options.architecture](sequence_shape, n_classes, activation='softmax', verbose=1)
            tmp_model.load_weights(model_path)
            models.append(tmp_model)
            del tmp_model

        res_save_path = os.path.join(result_run_dir, 'Ensemble.txt') if options.save_report else None
        report_paths.append(res_save_path.replace('.txt', '.pickle'))
        
        test_keras_ensemble(sess=sess, models=models, testX=testX, testY=testY, save_path=res_save_path)
        del models
        del testX, testY

    ###### Create mean eval reports #########
    if options.save_mcr:
        mean_report_path = os.path.join(result_dir, experiment_dirname, 'mean-report.txt')
        mean_cr_utils.compute_meanclassification_report(
            report_paths, save=mean_report_path, verbose=options.verbose, store_dict=True)


if __name__ == "__main__":
    # Command line options
    parser = OptionParser()

    ########## Global settings #############
    parser.add_option("--verbose", action="store_true",
                      dest="verbose", help="Flag to verbose")
    parser.add_option("--runs", action="store", type=int,
                      dest="runs", default=1, help="Number of runs to execute")
    parser.add_option("--architecture", action="store", type=str,
                      dest="architecture", default='AlexNet', help="AlexNet, LeNet, FCN, LSTM, FDN")

    ######### Dataset processing ###########
    parser.add_option("--root_path", action="store", type=str,
                      dest="root_path", default="../../../data/", help="Path that includes the different datasets")
    parser.add_option("--dataset_name", action="store", type=str,
                      dest="dataset_name", default="ElectricDevices", help="Name of the dataset folder")
    parser.add_option("--normalize", action="store_true",
                      dest="normalize", help="Flag to normalize the data")
    parser.add_option("--standardize", action="store_true",
                      dest="standardize", help="Flag to standardize the data")
    parser.add_option("--validation_split", action="store", type=float,
                      dest="validation_split", default=0.0, help="Creates a validation set, set to zero to exclude validation set")

    ######### Base parameter model #########
    parser.add_option("--load_model", action="store_true",
                      dest="load_model", help="Flag to load an existing model")
    parser.add_option("--save_model", action="store_true",
                      dest="save_model", help="Flag to save the model")
    parser.add_option("--epochs", action="store", type=int,
                      dest="epochs", default=100, help="Number of epochs")
    parser.add_option("--batch_size", action="store", type=int,
                      dest="batch_size", default=32, help="Batch size for training and prediction")
    parser.add_option("--learning_rate", action="store", type=float,
                      dest="learning_rate", default=0.2, help="Learning rate for training")

    ########## Privacy Parameter ###########
    parser.add_option("--l2_norm_clip", action="store", type=float,
                      dest="l2_norm_clip", default=1.0, help="Clipping value for the differential learning")
    parser.add_option("--noise_multiplier", action="store", type=float,
                      dest="noise_multiplier", default=0.1, help="Noise multiplier to make the data private")

    ########## Federated Parameters ########
    parser.add_option("--n_clients", action="store", type=int,
                      dest="n_clients", default=4, help="Flag to set number of clients")
    parser.add_option("--use_stratified", action="store_true",
                      dest="use_stratified", default=False, help="Flag split the data stratified to the owners")
    
    ############# Evaluation ###############
    parser.add_option("--save_report", action="store_true",
                      dest="save_report", help="Flag to save the evaluation report")
    parser.add_option("--save_mcr", action="store_true",
                      dest="save_mcr", help="Flag to save the mean evaluation report")

    # Parse command line options
    (options, args) = parser.parse_args()

    # print options
    print(options)

    process(options)
