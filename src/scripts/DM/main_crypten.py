from optparse import OptionParser

import numpy as np
import os
import random
import torch
import crypten
import crypten.mpc as mpc
import crypten.communicator as comm
import sys

############## Import modules ##############
sys.path.append("../../")

from modules.utils import compute_classification_report
from modules.AL.models.AlexNet2d import AlexNet2d as AlexNet
from modules.AL.utils import make_directory_if_not_exists
from modules import ucr_loader, utils

crypten.init()
torch.set_num_threads(1)

global n_clients
n_clients = 1

@mpc.run_multiprocess(world_size=n_clients)
def test(model, options, testX, testY, save_path, n_classes):
    dummy_size = (options.batch_size - (testX.shape[0] % options.batch_size)) % options.batch_size
    testXd = np.concatenate([testX, np.zeros(
        (dummy_size, *testX.shape[1:]))], axis=0) if dummy_size > 0 else testX
    testYd = np.concatenate([testY, np.zeros(dummy_size)],
                            axis=0) if dummy_size > 0 else testY

    rank = comm.get().get_rank()

    #######################
    ### Multiprocessing ###
    ###  & Encryption   ###
    #######################

    enc_splits = np.split(np.arange(testXd.shape[2]), options.n_clients)
    test_encs = [crypten.cryptensor(
        testXd[:, :, enc_splits[i], :], src=i) for i in range(options.n_clients)]
    testX_enc = crypten.cat(test_encs, dim=2)

    # One-hot encode labels
    label_eye = torch.eye(n_classes)
    testY_one = label_eye[testYd]

    # Test the model
    with torch.no_grad():
        model.eval()

        all_labels = []
        all_predicted = []

        for i in range(int(testX_enc.shape[0]/options.batch_size)):
            x = testX_enc[i*options.batch_size:(i+1)*options.batch_size]
            y = testY_one[i*options.batch_size:(i+1)*options.batch_size]

            outputs = model(x)

            if dummy_size > 0 and (i+1) == int(testX_enc.shape[0]/options.batch_size):
                outputs = outputs[:-dummy_size]
                y = y[:-dummy_size]

            _, predicted = torch.max(outputs.get_plain_text().data, 1)
            _, labels = torch.max(y.data, 1)

            all_labels.append(labels)
            all_predicted.append(predicted)

        all_labels = torch.cat(all_labels, dim=0).cpu().numpy().astype(int)
        all_predicted = torch.cat(
            all_predicted, dim=0).cpu().numpy().astype(int)

        if rank == 0:
            compute_classification_report(
                all_labels, all_predicted, verbose=1, save=save_path)


@mpc.run_multiprocess(world_size=n_clients)
def train(options, trainX, trainY, valX, valY, n_classes, save_path):
    dummy_size = (options.batch_size - (valX.shape[0] % options.batch_size)) % options.batch_size

    valXd = np.concatenate([valX, np.zeros(
        (dummy_size, *valX.shape[1:]))], axis=0) if dummy_size > 0 else valX
    valYd = np.concatenate([valY, np.zeros(dummy_size)],
                           axis=0) if dummy_size > 0 else valY

    rank = comm.get().get_rank()

    #######################
    ### Multiprocessing ###
    ###  & Encryption   ###
    #######################

    enc_splits = np.split(np.arange(trainX.shape[2]), options.n_clients)

    train_encs = [crypten.cryptensor(
        trainX[:, :, enc_splits[i], :], src=i) for i in range(options.n_clients)]
    trainX_enc = crypten.cat(train_encs, dim=2)

    val_encs = [crypten.cryptensor(
        valXd[:, :, enc_splits[i], :], src=i) for i in range(options.n_clients)]
    valX_enc = crypten.cat(val_encs, dim=2)

    # One-hot encode labels
    label_eye = torch.eye(n_classes)
    trainY_one = label_eye[trainY]
    valY_one = label_eye[valYd]

    sequence_length = trainX.shape[2]
    sequence_channels = trainX.shape[1]

    model_plain = AlexNet(in_width=sequence_length,
                          in_channels=sequence_channels, num_classes=n_classes)
    ### Encrypt Model ###
    model = crypten.nn.from_pytorch(
        model_plain, torch.empty(1, *trainX.shape[1:]))
    model.encrypt()

    criterion = crypten.nn.CrossEntropyLoss()

    # Train the model
    if rank == 0:
        crypten.print('\n--- Training Network ---')

    global_step = 0
    ES_counter = 0
    LR_counter = 0
    min_loss = sys.float_info.max
    current_lr = options.initial_lr
    current_val_loss = 0
    current_val_acc = 0
    current_train_acc = 0

    model.train()

    for epoch in range(options.epochs):

        correct = 0
        total = 0

        perm = np.random.permutation(trainX_enc.shape[0])
        shuffled_trainX_enc = trainX_enc[perm]
        shuffled_trainY_enc = trainY_one[perm]

        for i in range(int(trainX_enc.shape[0]/options.batch_size)):
            x = shuffled_trainX_enc[i *
                                    options.batch_size:(i+1)*options.batch_size]
            y = shuffled_trainY_enc[i *
                                    options.batch_size:(i+1)*options.batch_size]

            # Forward pass
            outputs = model(x)

            loss = criterion(outputs, y)

            # Backward and optimize
            model.zero_grad()
            loss.backward()
            model.update_parameters(current_lr)

            # Compute accuracy
            _, predicted = torch.max(outputs.get_plain_text().data, 1)
            _, labels = torch.max(y.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().cpu().data.numpy()

            global_step += 1

        current_train_acc = np.true_divide(correct, total)
        
        # Validate
        correct = 0
        total = 0

        with torch.no_grad():
            model.eval()
            for i in range(int(valX_enc.shape[0]/options.batch_size)):
                x = valX_enc[i*options.batch_size:(i+1)*options.batch_size]
                y = valY_one[i*options.batch_size:(i+1)*options.batch_size]

                # Forward pass
                outputs = model(x)

                if dummy_size > 0 and (i+1) == int(valX_enc.shape[0]/options.batch_size):
                    outputs = outputs[:-dummy_size]
                    y = y[:-dummy_size]

                # Compute accuracy
                _, predicted = torch.max(outputs.get_plain_text().data, 1)
                _, labels = torch.max(y.data, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().cpu().data.numpy()
                
                current_val_loss += criterion(outputs, y).get_plain_text().item()

            current_val_loss = current_val_loss / total
            current_val_acc = np.true_divide(correct, total)
            if rank == 0:
                crypten.print('Epoch [{}/{}], Step: {}, Loss: {:.4f}, Acc: {:.4f}, ESC: {}/{}, LR: {:.4f}, Val_Loss: {:.4f}, Val_Acc: {:.4f}'.format(
                    epoch + 1, options.epochs, global_step + 1,  loss.get_plain_text().item(), current_train_acc, ES_counter, options.es_patience, current_lr, current_val_loss, current_val_acc))

            # Check if early stopping applies
            if current_val_loss < min_loss:
                min_loss = current_val_loss
                ES_counter = 0
                LR_counter = 0
            else:
                if ES_counter >= options.es_patience:
                    break
                elif LR_counter >= options.lr_patience:
                    current_lr *= options.lr_multiplier

                    LR_counter = 0

                ES_counter += 1
                LR_counter += 1

    # Save Model
    torch.save(model.state_dict(), save_path)

def process(options):
    np.random.seed(options.seed)
    torch.manual_seed(0)

    DIR_RESULTS = '../../../results/'
    DIR_MODLES = '../../../models/'

    # get all datasets
    dataset_dict = ucr_loader.get_datasets(options.root_path, prefix='**/')
    # retrieve data
    dataset_name = options.dataset_name
    trainX, trainY, testX, testY = ucr_loader.load_data(
        dataset_dict[dataset_name])
    # preprocess data
    trainX, trainY, testX, testY = ucr_loader.preprocess_data(
        trainX, trainY, testX, testY, normalize=options.normalize, standardize=options.standardize)
    # additional preprocessing
    trainX, trainY, valX, valY = utils.perform_datasplit(
        trainX, trainY, test_split=options.validation_split)
    n_classes = len(np.unique(trainY))

    # Shapes
    print('TrainX:', trainX.shape)
    print('ValX:', valX.shape)
    print('TestX:', testX.shape)
    print('Classes:', n_classes)

    # Convert to channels first
    trainX = torch.tensor(np.expand_dims(trainX.transpose(0, 2, 1), axis=-1))
    trainY = torch.tensor(trainY)
    testX = torch.tensor(np.expand_dims(testX.transpose(0, 2, 1), axis=-1))
    testY = torch.tensor(testY)
    valX = torch.tensor(np.expand_dims(valX.transpose(0, 2, 1), axis=-1))
    valY = torch.tensor(valY)

    model_name = 'model.ckpt'
    model_directory = 'CrypTen_AlexNet_' + dataset_name
    experiment_directory = 'BS_%s_LR_%.4f_NWorlds_%s' % (
        options.batch_size, options.initial_lr, options.n_clients)
    model_save_path = os.path.join(
        DIR_MODLES, model_directory, experiment_directory)
    make_directory_if_not_exists(model_save_path)
    model_save_path = os.path.join(model_save_path, model_name)

    filename = 'report.txt'
    res_save_path = os.path.join(
        DIR_RESULTS, model_directory, experiment_directory)
    make_directory_if_not_exists(res_save_path)
    res_save_path = os.path.join(res_save_path, filename)
    if not os.path.isfile(model_save_path):
        train(options=options,
                      trainX=trainX,
                      trainY=trainY,
                      valX=valX,
                      valY=valY,
                      n_classes=n_classes,
                      save_path=model_save_path)
    
    # load model
    print('Trained model found: ', model_save_path)
    model_plain = AlexNet(in_width=trainX.shape[2],
                        in_channels=trainX.shape[1], num_classes=n_classes)
    model = crypten.nn.from_pytorch(
        model_plain, torch.empty(1, *trainX.shape[1:]))
    model.encrypt()
    model.load_state_dict(torch.load(model_save_path))

    # Test Model
    test(model=model,
         options=options,
         testX=testX,
         testY=testY,
         save_path=res_save_path,
         n_classes=n_classes)


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
    parser.add_option("--num_cpus", action="store", type=int,
                      dest="num_cpus", default=11, help="Number of cpus to use for preprocessing")

    ####### Perform baseline model #########
    parser.add_option("--load_model", action="store_true",
                      dest="load_model", help="Flag to load an existing model")
    parser.add_option("--save_model", action="store_true",
                      dest="save_model", help="Flag to save the model")
    parser.add_option("--epochs", action="store", type=int,
                      dest="epochs", default=100, help="Number of epochs")
    parser.add_option("--batch_size", action="store", type=int,
                      dest="batch_size", default=32, help="Batch size for training and prediction")
    parser.add_option("--initial_lr", action="store", type=float,
                      dest="initial_lr", default=0.01, help="Initial Learning Rate")
    parser.add_option("--lr_patience", action="store", type=int,
                      dest="lr_patience", default=5, help="Patience for LR decay")
    parser.add_option("--lr_multiplier", action="store", type=float,
                      dest="lr_multiplier", default=0.5, help="Factor for LR decay")
    parser.add_option("--es_patience", action="store", type=int,
                      dest="es_patience", default=10, help="Patience for Early Stopping")

    ########## Federated Parameter #########
    parser.add_option("--n_clients", action="store", type=int,
                      dest="n_clients", default=4, help="Flag to set number of clients")

    # Parse command line options
    (options, args) = parser.parse_args()

    torch.manual_seed(options.seed)
    random.seed(options.seed)
    np.random.seed(options.seed)

    # print options
    print(options)

    process(options)
