import torch
import numpy as np
from sklearn.naive_bayes import GaussianNB

import sys

############## Import modules ##############
sys.path.append("../../")

from modules.utils import compute_classification_report

def test_torch(model, test_loader, device, save_path=False, return_accuracy=False):
    with torch.no_grad():
        model.to(device)
        model.eval()

        all_labels = []
        all_predicted = []

        for step, (x, y) in enumerate(test_loader):
            
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            outputs = model(x.float())
            
            _, predicted = torch.max(outputs.data, 1)

            all_labels.append(y)
            all_predicted.append(predicted)

        all_labels = torch.cat(all_labels, dim=0).cpu().numpy().astype(int)
        all_predicted = torch.cat(all_predicted, dim=0).cpu().numpy().astype(int)

        if not return_accuracy:
            compute_classification_report(all_labels, all_predicted, verbose=1, save=save_path, store_dict=False)
        else:
            res = np.mean(all_predicted == all_labels)

            return res

def test_torch_ensemble(models, test_loader, device, ensemble_method='average_output', ensemble_weights=None, save_path=False):
    with torch.no_grad():
        for model in models:
            model.to(device)
            model.eval()

        all_labels = []
        all_predicted = []

        for step, (x, y) in enumerate(test_loader):
            
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            outputs = [model(x.float()).cpu().numpy() for model in models]

            weights = [1.0/len(models) for _ in range(len(models))] if ensemble_weights is None else ensemble_weights

            if ensemble_method == 'average_output':
                outputs = np.array(outputs)
                outputs = np.tensordot(outputs, weights, axes=((0), (0)))
                outputs = torch.tensor(outputs)
                outputs = outputs.to(device)
                _, predicted = torch.max(outputs.data, 1)
            elif ensemble_method == 'majority_vote':
                predictions = [np.argmax(o, axis=1) for o in outputs]
                predictions = np.array(predictions)
                
                predicted = []
                # for each sample
                for i in range(predictions.shape[-1]):
                    tmp_labels, tmp_counts = np.unique(predictions[:,i], return_counts=True)
                    predicted.append(tmp_labels[0])

                predicted = torch.tensor(predicted)
                predicted = predicted.to(device)

            all_labels.append(y)
            all_predicted.append(predicted)

        all_labels = torch.cat(all_labels, dim=0).cpu().numpy().astype(int)
        all_predicted = torch.cat(all_predicted, dim=0).cpu().numpy().astype(int)

        report = compute_classification_report(all_labels, all_predicted, verbose=1, save=save_path, store_dict=True)

def train_test_nb_model(models, train_loader, test_loader, device, save_path=False):

    with torch.no_grad():
        for model in models:
            model.to(device)
            model.eval()


        # Train NB classifier
        all_labels = []
        all_outputs = []

        for step, (x, y) in enumerate(train_loader):
            
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            outputs = [model(x.float()).cpu().numpy() for model in models]
            outputs = np.concatenate(outputs, axis=-1)

            all_labels.append(y)
            all_outputs.append(outputs)

        all_labels = torch.cat(all_labels, dim=0).cpu().numpy().astype(int)
        all_outputs = np.concatenate(all_outputs, axis=0)

        model_sk = GaussianNB(priors = None)
        model_sk.fit(all_outputs, all_labels)

        # Test NB classifier
        all_labels = []
        all_outputs = []

        for step, (x, y) in enumerate(test_loader):
            
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            outputs = [model(x.float()).cpu().numpy() for model in models]
            outputs = np.concatenate(outputs, axis=-1)

            all_labels.append(y)
            all_outputs.append(outputs)

        all_labels = torch.cat(all_labels, dim=0).cpu().numpy().astype(int)
        all_outputs = np.concatenate(all_outputs, axis=0)

        nb_predictions = model_sk.predict(all_outputs)

        print('Classification Report Naive Bayes:')
        report = compute_classification_report(all_labels, nb_predictions, verbose=1, save=save_path, store_dict=True)

        
