import os
import sys
from optparse import OptionParser

import numpy as np
import tensorflow as tf

############## Import modules ##############
sys.path.append("../")
from modules.model_definition import FCN, FDN, LSTM, AlexNet, LeNet
from modules import mean_cr_utils, model_trainer, ucr_loader, utils


def process(options):
    ########## Global settings #############
    np.random.seed(options.seed)
    model_dir, result_dir = utils.maybe_create_dirs(
        options.dataset_name, root='../../', dirs=['models', 'results'], exp=options.exp_path, return_paths=True, verbose=options.verbose)

    ######### Dataset processing ###########
    dataset_dict = ucr_loader.get_datasets(options.root_path, prefix='**/')
    trainX, trainY, testX, testY = ucr_loader.load_data(
        dataset_dict[options.dataset_name])
    trainX, trainY, testX, testY = ucr_loader.preprocess_data(
        trainX, trainY, testX, testY, normalize=options.normalize, standardize=options.standardize)
    valX, valY = None, None
    n_classes = len(np.unique(trainY))

    if options.validation_split > 0:
        trainX, trainY, valX, valY = utils.perform_datasplit(
            trainX, trainY, test_split=options.validation_split)
    if options.verbose:
        print('TrainX:', trainX.shape)
        if options.validation_split > 0:
            print('ValX:', valX.shape)
        print('TestX:', testX.shape)
        print('Classes:', n_classes)

    ##### model architecture ######
    architecture_func = {'AlexNet': AlexNet().build_1d, 'LSTM': LSTM().build_default,
                         'FCN': FCN().build_default, 'FDN': FDN().build_default, 'LeNet': LeNet().build_default}

    report_paths = []
    for i in range(options.runs):
        tf.random.set_seed(i)
        
        if options.verbose:
            print('Run %d / %d' % (i+1, options.runs))
        ####### Perform baseline model #########
        model_path = os.path.join(model_dir, options.architecture + '_batch-' + str(
            options.batch_size) + '_run-' + str(i) + '.h5') if options.save_model else None

        if os.path.exists(model_path) and options.load_model:
            model = architecture_func[options.architecture](
                trainX.shape[1:], n_classes, activation='softmax', verbose=options.verbose)
            model.load_weights(model_path)
        else:
            model = architecture_func[options.architecture](
                trainX.shape[1:], n_classes, activation='softmax', verbose=options.verbose)
            model.compile(
                optimizer='SGD', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            model_trainer.train(model, trainX, trainY, validation_data=(
                valX, valY), epochs=options.epochs, batch_size=options.batch_size, model_path=model_path, verbose=options.verbose)

        ############# Evaluation ###############
        report_path = os.path.join(result_dir, options.architecture + '_batch-' + str(
            options.batch_size) + '_run-' + str(i) + '_report.txt') if options.save_report else None
        preds = np.argmax(model.predict(
            testX, batch_size=options.batch_size, verbose=options.verbose), axis=1)
        utils.compute_classification_report(
            testY, preds, save=report_path, verbose=options.verbose, store_dict=True)
        report_paths.append(report_path.replace('.txt', '.pickle'))

    ###### Create mean eval report #########
    if options.save_mcr:
        mean_report_path = os.path.join(result_dir, options.architecture + '_batch-' + str(
            options.batch_size) + '_mean-report.txt')
        mean_cr_utils.compute_meanclassification_report(
            report_paths, save=mean_report_path, verbose=options.verbose, store_dict=True)


if __name__ == "__main__":
    # Command line options
    parser = OptionParser()

    ########## Global settings #############
    parser.add_option("--verbose", action="store_true",
                      dest="verbose", help="Flag to verbose")
    parser.add_option("--seed", action="store", type=int,
                      dest="seed", default=0, help="random seed")

    ######### Dataset processing ###########
    parser.add_option("--root_path", action="store", type=str,
                      dest="root_path", default="../../data/", help="Path that includes the different datasets")
    parser.add_option("--dataset_name", action="store", type=str,
                      dest="dataset_name", default="ElectricDevices", help="Name of the dataset folder")
    parser.add_option("--normalize", action="store_true",
                      dest="normalize", help="Flag to normalize the data")
    parser.add_option("--standardize", action="store_true",
                      dest="standardize", help="Flag to standardize the data")
    parser.add_option("--validation_split", action="store", type=float,
                      dest="validation_split", default=0.0, help="Creates a validation set, set to zero to exclude validation set")

    ######### Experiment details ###########
    parser.add_option("--runs", action="store", type=int,
                      dest="runs", default=1, help="Number of runs to execute")
    parser.add_option("--exp_path", action="store", type=str,
                      dest="exp_path", default=None, help="Sub-Folder for experiment setup")
    parser.add_option("--architecture", action="store", type=str,
                      dest="architecture", default='AlexNet', help="AlexNet, LeNet, FCN, LSTM, FDN")

    ####### Perform baseline model #########
    parser.add_option("--load_model", action="store_true",
                      dest="load_model", help="Flag to load an existing model")
    parser.add_option("--save_model", action="store_true",
                      dest="save_model", help="Flag to save the model")
    parser.add_option("--epochs", action="store", type=int,
                      dest="epochs", default=100, help="Number of epochs")
    parser.add_option("--batch_size", action="store", type=int,
                      dest="batch_size", default=8, help="Batch size for training and prediction")

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
