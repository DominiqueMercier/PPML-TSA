import os
import pickle

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


def perform_datasplit(data, labels, test_split=0.3, stratify=True, return_state=False, random_state=0):
    try:
        da, db, la, lb = train_test_split(
            data, labels, test_size=test_split, random_state=random_state, stratify=labels if stratify else None)
        state = True
    except:
        da, db, la, lb = train_test_split(
            data, labels, test_size=test_split, random_state=random_state, stratify=None)
        print('Warining: No stratified split possible')
        state = False
    if return_state:
        return da, la, db, lb, state
    return da, la, db, lb


def compute_classification_report(gt, preds, save=False, verbose=0, store_dict=False):
    s = classification_report(gt, preds, digits=4)
    if verbose:
        print(s)
    if not save is None:
        with open(save, 'w') as f:
            f.write(s)
        if verbose:
            print('Save Location:', save)
        if store_dict:
            cr_dict = classification_report(gt, preds, digits=4, output_dict=True)
            with open(save.replace('.txt', '.pickle'), 'wb') as f:
                pickle.dump(cr_dict, f)


def maybe_create_dirs(dataset_name, root='../../', dirs=['models', 'results'], exp=None, return_paths=False, verbose=0):
    paths = []
    for d in dirs:
        if exp is None:
            tmp = os.path.join(root, d, dataset_name)
        else:
            tmp = os.path.join(root, d, exp, dataset_name)
        paths.append(tmp)
        if not os. path.exists(tmp):
            os.makedirs(tmp)
            if verbose:
                print('Created directory:', tmp)
        elif verbose:
            print('Found existing directory:', tmp)
    if return_paths:
        return paths
