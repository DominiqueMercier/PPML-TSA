import os
import sys
from optparse import OptionParser

import numpy as np
import tensorflow as tf
from tensorflow_privacy.privacy.analysis import compute_dp_sgd_privacy
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import \
    DPKerasSGDOptimizer

############ Tensforflow config ############
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
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

    #### Prepare differential datasets #####
    train_size, val_size = int(trainX.shape[0]/options.batch_size) * options.batch_size, int(
        valX.shape[0]/options.batch_size) * options.batch_size
    trainXm, trainYm, valXm, valYm = trainX[:train_size], trainY[:
                                                                 train_size], valX[:val_size], valY[:val_size]

    ##### model architecture ######
    architecture_func = {'AlexNet': AlexNet().build_1d, 'LSTM': LSTM().build_default,
                         'FCN': FCN().build_default, 'FDN': FDN().build_default, 'LeNet': LeNet().build_default}

    report_paths = []
    for i in range(options.runs):
        tf.random.set_seed(i)
        
        if options.verbose:
            print('Run %d / %d' % (i+1, options.runs))
        ###### Train differential model ########
        model_path = os.path.join(model_dir, options.architecture + '_differential_1d_batch-' + str(options.batch_size) + '_l2-' + str(options.l2_norm_clip) \
            + '_noise-' + str(options.noise_multiplier) + '_run-' + str(i) + '.h5') if options.save_model else None
        if os.path.exists(model_path) and options.load_model:
            model = architecture_func[options.architecture](
                trainX.shape[1:], n_classes, activation='softmax', verbose=1)
            model.load_weights(model_path)
        else:
            model = architecture_func[options.architecture](
                trainX.shape[1:], n_classes, activation='softmax', verbose=1)
            optimizer = DPKerasSGDOptimizer(l2_norm_clip=options.l2_norm_clip, noise_multiplier=options.noise_multiplier,
                                            num_microbatches=options.batch_size, learning_rate=options.learning_rate)
            loss = tf.keras.losses.SparseCategoricalCrossentropy(
                from_logits=True, reduction=tf.losses.Reduction.NONE)
            model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
            trainer_differential_privacy.traintf(sess, model, trainXm, trainYm, validation_data=(valXm, valYm),
                                                epochs=options.epochs, batch_size=options.batch_size, model_path=model_path, verbose=options.verbose)

        ############# Evaluation ###############
        test_size = int(testX.shape[0]/options.batch_size) * options.batch_size
        testXm = testX
        if test_size < testX.shape[0]:
            fill = np.zeros(
                (options.batch_size - (testX.shape[0] - test_size), *testX.shape[1:]))
            testXm = np.concatenate([testX, fill], axis=0)

        report_path = os.path.join(
            result_dir, options.architecture + '_differential_1d_batch-' + str(options.batch_size) + '_l2-' + str(options.l2_norm_clip) + '_noise-' + str(options.noise_multiplier) \
                + '_run-' + str(i) + '_report.txt') if options.save_report else None
        preds = np.argmax(model.predict(testXm, batch_size=options.batch_size,
                                        verbose=options.verbose), axis=1)
        utils.compute_classification_report(
            testY, preds[:testX.shape[0]], save=report_path, verbose=options.verbose, store_dict=True)
        report_paths.append(report_path.replace('.txt', '.pickle'))
    
    ###### Create mean eval report #########
    if options.save_mcr:
        mean_report_path = os.path.join(result_dir, options.architecture + '_differential_1d_batch-' + str(
            options.batch_size) + '_l2-' + str(options.l2_norm_clip) + '_noise-' + str(options.noise_multiplier) + '_mean-report.txt')
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
    parser.add_option("--learning_rate", action="store", type=float,
                      dest="learning_rate", default=0.2, help="Learning rate for training")

    ########## Privacy Parameter ###########
    parser.add_option("--l2_norm_clip", action="store", type=float,
                      dest="l2_norm_clip", default=1.0, help="Clipping value for the differential learning")
    parser.add_option("--noise_multiplier", action="store", type=float,
                      dest="noise_multiplier", default=0.1, help="Noise multiplier to make the data private")
    
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
