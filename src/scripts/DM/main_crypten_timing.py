from optparse import OptionParser
from time import time

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

from modules.AL.dataloader import GenericDataset
from modules.AL.pt_utils import test_torch, test_torch_ensemble
from modules.AL.models.AlexNet2d import AlexNet2d as AlexNet
from modules.AL.utils import make_directory_if_not_exists
from modules import ucr_loader, utils


crypten.init()
torch.set_num_threads(1)


def train_crpyten(options, model, trainX, trainY, n_classes):
    trainX_enc = crypten.cryptensor(trainX, src=0)
    # One-hot encode labels
    label_eye = torch.eye(n_classes)
    trainY_one = label_eye[trainY]
    perm = np.random.permutation(trainX_enc.shape[0])
    shuffled_trainX_enc = trainX_enc[perm]
    shuffled_trainY_enc = trainY_one[perm]
    criterion = crypten.nn.CrossEntropyLoss()
    model.train()
    timer_start = time()
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
        model.update_parameters(options.initial_lr)
    overall_time = time() - timer_start
    return overall_time


def eval_crpyten(options, model, valX, valY, n_classes):
    dummy_size = (options.batch_size -
                  (valX.shape[0] % options.batch_size)) % options.batch_size
    valXd = np.concatenate([valX, np.zeros(
        (dummy_size, *valX.shape[1:]))], axis=0) if dummy_size > 0 else valX
    valYd = np.concatenate([valY, np.zeros(dummy_size)],
                           axis=0) if dummy_size > 0 else valY
    valX_enc = crypten.cryptensor(valXd, src=0)
    # One-hot encode labels
    label_eye = torch.eye(n_classes)
    valY_one = label_eye[valYd]
    with torch.no_grad():
        model.eval()
        timer_start = time()
        for i in range(int(valX_enc.shape[0]/options.batch_size)):
            x = valX_enc[i*options.batch_size:(i+1)*options.batch_size]
            y = valY_one[i*options.batch_size:(i+1)*options.batch_size]
            # Forward pass
            outputs = model(x)
            if dummy_size > 0 and (i+1) == int(valX_enc.shape[0]/options.batch_size):
                outputs = outputs[:-dummy_size]
                y = y[:-dummy_size]
        overall_time = time() - timer_start
    return overall_time


def train_pytorch(options, model, train_loader, use_gpu=True):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=options.initial_lr)
    device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
    # Train the model
    model.to(device)
    model.train()
    timer_start = time()
    for step, (x, y) in enumerate(train_loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        # Forward pass
        outputs = model(x.float())
        loss = criterion(outputs, y)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    overall_time = time() - timer_start
    return overall_time


def eval_pytorch(options, model, val_loader, use_gpu=True):
    device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
    model.to(device)
    with torch.no_grad():
        model.eval()
        timer_start = time()
        for step, (x, y) in enumerate(val_loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            # Forward pass
            outputs = model(x.float())
        overall_time = time() - timer_start
    return overall_time


def process(options):

    DIR_RESULTS = '../../../results/'

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

    if 1:  # options.only_batch:
        trainX = trainX[:options.batch_size]
        trainY = trainY[:options.batch_size]
        valX = valX[:options.batch_size]
        valY = valY[:options.batch_size]

    # Shapes
    print('TrainX:', trainX.shape)
    print('ValX:', valX.shape)
    print('TestX:', testX.shape)
    print('Classes:', n_classes)

    # Convert to channels first
    trainX = torch.tensor(np.expand_dims(trainX.transpose(0, 2, 1), axis=-1))
    trainY = torch.tensor(trainY)
    valX = torch.tensor(np.expand_dims(valX.transpose(0, 2, 1), axis=-1))
    valY = torch.tensor(valY)

    # get data loader
    train_dataset = GenericDataset(x=trainX, y=trainY)
    val_dataset = GenericDataset(x=valX, y=valY)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=options.batch_size,
                                               shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                             batch_size=options.batch_size)

    # create plain
    model_plain = AlexNet(in_width=trainX.shape[2],
                          in_channels=trainX.shape[1], num_classes=n_classes)
    model = crypten.nn.from_pytorch(
        model_plain, torch.empty(1, *trainX.shape[1:]))
    model.encrypt()

    times = np.zeros((6, options.trys))
    for i in range(options.trys):
        print('Process run: %s / %s' % (i+1, options.trys))
        print('CrypTen...', end='')
        times[0, i] = train_crpyten(options=options, model=model,
                            trainX=trainX, trainY=trainY, n_classes=n_classes)
        times[1, i] = eval_crpyten(options=options, model=model,
                        valX=valX, valY=valY, n_classes=n_classes)
        print('PytorchCPU...', end='')
        times[2, i] = train_pytorch(options=options, model=model_plain,
                            train_loader=train_loader, use_gpu=False)
        times[3, i] = eval_pytorch(options=options, model=model_plain,
                           val_loader=val_loader, use_gpu=False)
        print('PytorchGPU...', end='')
        times[4, i] = train_pytorch(options=options, model=model_plain,
                            train_loader=train_loader, use_gpu=True)
        times[5, i] = eval_pytorch(options=options, model=model_plain,
                           val_loader=val_loader, use_gpu=True)
        print('Finished')
    times *= 1000 # milli seconds
    
    model_directory = 'CrypTen_AlexNet_' + dataset_name
    experiment_directory = 'BS_%s_Trys_%s' % (
        options.batch_size, options.trys)

    filename = 'report.txt'
    res_save_path = os.path.join(
        DIR_RESULTS, model_directory, experiment_directory)
    make_directory_if_not_exists(res_save_path)
    res_save_path = os.path.join(res_save_path, filename)

    approaches = ['TrainCrypten', 'InferCrypten', 'TrainPytorchCPU', 'InferPytorchCPU', 'TrainPytorchGPU', 'InferPytorchGPU'] 
    s = 'Times in milliseconds\n'
    for vals, name in zip(times, approaches):
            s += '%s: %s | Std: %s | Min: %s |Max: %s\n' % (name, np.average(vals), np.std(vals), np.min(vals), np.max(vals))
    if options.verbose:
        print(s)
    with open(res_save_path, "w") as f:
        f.write(s)

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
    parser.add_option("--batch_size", action="store", type=int,
                      dest="batch_size", default=32, help="Batch size for training and prediction")
    parser.add_option("--initial_lr", action="store", type=float,
                      dest="initial_lr", default=0.01, help="Initial Learning Rate")
    parser.add_option("--trys", action="store", type=int,
                      dest="trys", default=10, help="Number of tries to average")
    
    # Parse command line options
    (options, args) = parser.parse_args()

    torch.manual_seed(options.seed)
    random.seed(options.seed)
    np.random.seed(options.seed)

    # print options
    print(options)

    process(options)
