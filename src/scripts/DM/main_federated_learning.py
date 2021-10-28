import os
import sys
from optparse import OptionParser

import nest_asyncio
import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

nest_asyncio.apply()

############ Tensforflow config ############
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

############## Import modules ##############
sys.path.append("../../")
from modules.DM import trainer_federated
from modules import ucr_loader, utils, mean_cr_utils
from modules.model_definition import FCN, FDN, LSTM, AlexNet, LeNet


def process(options):
    ########## Global settings #############
    np.random.seed(options.seed)
    model_dir, result_dir = utils.maybe_create_dirs(
        options.dataset_name, root='../../../', dirs=['models', 'results'], exp=options.exp_path, return_paths=True, verbose=options.verbose)
    
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
        if options.verbose:
            print('Run %d / %d' % (i+1, options.runs))
        ###### Prepare federated datasets ######
        model_path = os.path.join(model_dir, options.architecture + '_1d_federated_batch-' + str(options.batch_size) + '_stratified-' + str(options.use_stratified) \
            + '_nclients-' + str(options.n_clients) + '_nparallel-' + str(options.parallel_clients) + '_adaptive-' + str(options.adaptive) \
            + '_run-' + str(i) + '.h5') if options.save_model else None
        if options.stepwise and options.save_model:
            model_path = model_path.replace('_run-' + str(i) + '.h5', '_stepwise-True_run-' + str(i) + '.h5')
        stratified_state = options.use_stratified
        if not os.path.exists(model_path) or not options.load_model:
            federated_train_pre, stratified_state = trainer_federated.make_federated_data(
                trainX, trainY, n_clients=options.n_clients, stratify=options.use_stratified, return_state=True, random_state=i)
            federated_val_pre = trainer_federated.make_federated_data(
                valX, valY, n_clients=options.n_clients, stratify=options.use_stratified, random_state=i)
            federated_train = trainer_federated.preporcess_federated_data(
                federated_train_pre, num_epochs=options.client_epochs, batch_size=options.batch_size)
            federated_val = trainer_federated.preporcess_federated_data(
                federated_val_pre, num_epochs=1, batch_size=options.batch_size)
            if stratified_state != options.use_stratified:
                model_path = os.path.join(model_dir, options.architecture + '_1d_federated_batch-' + str(options.batch_size) + '_stratified-' + str(stratified_state) \
                    + '_nclients-' + str(options.n_clients) + '_nparallel-' + str(options.parallel_clients) + '_adaptive-' + str(options.adaptive) \
                    + '_run-' + str(i) + '.h5') if options.save_model else None
                if options.stepwise and options.save_model:
                    model_path = model_path.replace('_run-' + str(i) + '.h5', '_stepwise-True_run-' + str(i) + '.h5')

        ####### Train federated model ##########
        if os.path.exists(model_path) and options.load_model:
            if options.architecture == 'AlexNet':
                model_extern = architecture_func[options.architecture](trainX.shape[1:], n_classes, batch_norm=False)
            else:
                model_extern = architecture_func[options.architecture](trainX.shape[1:], n_classes)
            model_extern.load_weights(model_path)
        else:
            # definition
            federated_dict = {}
            federated_dict['input_shape'] = trainX.shape[1:]
            federated_dict['n_classes'] = n_classes
            federated_dict['input_spec'] = federated_train[0].element_spec
            federated_dict['architecture'] = options.architecture
            federated_dict['architecture_func'] = architecture_func

            federated_train_process, federated_eval_process = trainer_federated.create_federated_processes(federated_dict, options.server_learning_rate, 
                client_learning_rate=options.client_learning_rate, adaptive=options.adaptive, stepwise=options.stepwise)

            state = federated_train_process.initialize()
            # training
            model_extern = trainer_federated.traintf(state, federated_dict, federated_train_process, federated_train_pre,
                                                    federated_eval_process=federated_eval_process, federated_val=federated_val,
                                                    n_clients=options.parallel_clients, epochs=options.epochs, batch_size=options.batch_size, 
                                                    lr_decay=options.adaptive, model_path=model_path, verbose=options.verbose)

        ############# Evaluation ###############
        report_path = os.path.join(result_dir, options.architecture + '_federated_batch-' + str(options.batch_size) + '_stratified-' + str(stratified_state) \
            + '_nclients-' + str(options.n_clients) + '_nparallel-' + str(options.parallel_clients) \
            + '_adaptive-' + str(options.adaptive) + '_run-' + str(i) + '_report.txt') if options.save_report else None
        if options.stepwise:
            report_path = report_path.replace('_run-' + str(i) + '.txt', '_stepwise-True' + '_run-' + str(i) + '.txt')
        preds = np.argmax(model_extern.predict(
            testX, batch_size=options.batch_size, verbose=options.verbose), axis=1)
        utils.compute_classification_report(
            testY, preds, save=report_path, verbose=options.verbose, store_dict=True)
        report_paths.append(report_path.replace('.txt', '.pickle'))

    ###### Create mean eval report #########
    if options.save_mcr:
        mean_report_path = os.path.join(result_dir, options.architecture + '_federated_batch-' + str(options.batch_size) + '_stratified-' + str(stratified_state) \
            + '_nclients-' + str(options.n_clients) + '_nparallel-' + str(options.parallel_clients) \
            + '_adaptive-' + str(options.adaptive) + '_mean-report.txt')
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
                      dest="root_path", default="../../../data/", help="Path that includes the different datasets")
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

    ######### Base parameter model #########
    parser.add_option("--load_model", action="store_true",
                      dest="load_model", help="Flag to load an existing model")
    parser.add_option("--save_model", action="store_true",
                      dest="save_model", help="Flag to save the model")
    parser.add_option("--epochs", action="store", type=int,
                      dest="epochs", default=100, help="Number of epochs")
    parser.add_option("--batch_size", action="store", type=int,
                      dest="batch_size", default=32, help="Batch size for training and prediction")
    
    ########## Federated Parameter #########
    parser.add_option("--client_epochs", action="store", type=int,
                      dest="client_epochs", default=1, help="Number of epochs for each client before aggregation")
    parser.add_option("--n_clients", action="store", type=int,
                      dest="n_clients", default=4, help="Flag to set number of clients")
    parser.add_option("--parallel_clients", action="store", type=int,
                      dest="parallel_clients", default=None, help="Flag to define parallel clients during training")
    parser.add_option("--use_stratified", action="store_true",
                      dest="use_stratified", default=False, help="Flag split the data stratified to the owners")
    parser.add_option("--adaptive", action="store_true",
                      dest="adaptive", default=False, help="Flag for adaptive learning rate schedule")
    parser.add_option("--stepwise", action="store_true",
                      dest="stepwise", help="Flag for stepwise process e.g. train local step instead of local epochs")
    parser.add_option("--server_learning_rate", action="store", type=float,
                      dest="server_learning_rate", default=0.2, help="Learning rate for training")
    parser.add_option("--client_learning_rate", action="store", type=float,
                      dest="client_learning_rate", default=0.02, help="Learning rate for training")

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
