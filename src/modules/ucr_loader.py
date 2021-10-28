import os
from glob import glob

import numpy as np
import sktime
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sktime.utils.data_io import (load_from_arff_to_dataframe,
                                  load_from_tsfile_to_dataframe,
                                  load_from_ucr_tsv_to_dataframe)
from sktime.utils.data_processing import from_nested_to_3d_numpy


def get_datasets(root, prefix='**/**/'):
    flist = glob(root + prefix)
    ddict = {}
    for f in flist:
        ddict[f.split('/')[-2]] = f
    return ddict

def to_array(df):
    arr = df.values.tolist()
    ma = np.max([len(a[0]) for a in arr])
    result = np.empty((*df.shape, ma))
    result[:] = np.NaN
    for i, d in enumerate(arr):
        for c in range(len(d)):
            result[i,c,:len(d[c])] = d[c]
    return result

def load_data(path):
    fname = path.split('/')[-2]
    train_path = os.path.join(path, fname + '_TRAIN.ts')
    test_path = os.path.join(path, fname + '_TEST.ts')
    if os.path.exists(train_path):
        train_x, train_y = load_from_tsfile_to_dataframe(
            train_path, replace_missing_vals_with='NaN')
        test_x, test_y = load_from_tsfile_to_dataframe(
            test_path, replace_missing_vals_with="NaN")
    elif os.path.exists(train_path.replace('_TRAIN.ts', '_TRAIN.arff')):
        train_x, train_y = load_from_arff_to_dataframe(
            train_path.replace('_TRAIN.ts', '_TRAIN.arff'), replace_missing_vals_with="NaN")
        test_x, test_y = load_from_arff_to_dataframe(
            test_path.replace('_TEST.ts', '_TEST.arff'), replace_missing_vals_with="NaN")
    else:
        train_x, train_y = load_from_ucr_tsv_to_dataframe(
            train_path.replace('_TRAIN.ts', '_TRAIN.tsv'))
        test_x, test_y = load_from_ucr_tsv_to_dataframe(
            test_path.replace('_TEST.ts', '_TEST.tsv'))
    # convert to numpy
    try:
        train_x_tmp = from_nested_to_3d_numpy(train_x).transpose(0, 2, 1)
        test_x_tmp = from_nested_to_3d_numpy(test_x).transpose(0, 2, 1)
        train_x = train_x_tmp
        test_x = test_x_tmp
    except:
        train_x = to_array(train_x).transpose(0,2,1)
        test_x = to_array(test_x).transpose(0,2,1)

        max_ts = np.max([train_x.shape[1], test_x.shape[1]])
        def maybe_append(data, tsize):
            if data.shape[1] < tsize:
                fill_arr = np.empty((data.shape[0], tsize-data.shape[1], data.shape[2]))
                fill_arr[:] = np.NaN
                return np.concatenate([train_x, fill_arr], axis=1)
            return data
        
        train_x = maybe_append(train_x, max_ts)
        test_x = maybe_append(test_x, max_ts)
    # shorten testset size if too long
    test_x = test_x[:, :train_x.shape[1], :]

    return train_x, train_y, test_x, test_y


def scale_data(trainX, testX, mode='standardize'):
    scaler = StandardScaler() if mode == 'standardize' else MinMaxScaler()
    org_train_shape = trainX.shape[1:]
    org_test_shape = testX.shape[1:]
    trainXf = trainX.reshape(-1, org_train_shape[-1])
    testXf = testX.reshape(-1, org_test_shape[-1])
    scaler.fit(trainXf)
    trainX = scaler.transform(trainXf).reshape(-1, *org_train_shape)
    testX = scaler.transform(testXf).reshape(-1, *org_test_shape)
    return trainX, testX


def preprocess_data(trainX, trainY, testX, testY, normalize=False, standardize=True):
    # adjust labels
    le = LabelEncoder().fit(trainY)
    trainY = le.transform(trainY)
    testY = le.transform(testY)
    # remove missings
    cmean = np.nanmean(trainX, axis=(0, 1))
    inds = np.where(np.isnan(trainX))
    trainX[inds] = np.take(cmean, inds[2])
    inds = np.where(np.isnan(testX))
    testX[inds] = np.take(cmean, inds[2])
    # standardize
    if standardize:
        trainX, testX = scale_data(trainX, testX, mode='standardize')
    # normalize
    if normalize:
        trainX, testX = scale_data(trainX, testX, mode='normalize')
    return trainX, trainY, testX, testY
