import sys
sys.path.append("../../")

import os
from modules import utils

def make_federated_data(data, labels, n_clients=2, stratify=True):
    split_size = 1 / n_clients
    divisor = 1
    restX, restY = data, labels
    federated_X = []
    federated_Y = []
    for _ in range(n_clients-1):
        if stratify:
            restX, restY, tmpX, tmpY = utils.perform_datasplit(
                restX, restY, test_split=split_size / divisor, stratify=True)
        else:
            restX, restY, tmpX, tmpY = utils.perform_datasplit(
                restX, restY, test_split=split_size / divisor, stratify=False)
        divisor -= split_size
        federated_X.append(tmpX)
        federated_Y.append(tmpY)
    federated_X.append(restX)
    federated_Y.append(restY)
    return federated_X, federated_Y 

def make_directory_if_not_exists(path):

    if not os.path.exists(path):
        os.makedirs(path)

