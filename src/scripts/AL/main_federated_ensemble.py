from optparse import OptionParser

import numpy as np
import os
import random
import torch
import sys

############## Import modules ##############
sys.path.append("../../")

from modules import ucr_loader, utils, mean_cr_utils
from modules.AL.dataloader import GenericDataset
from modules.AL.utils import make_federated_data, make_directory_if_not_exists
from modules.AL.models.AlexNet1d import AlexNet1d as AlexNet
from modules.AL.models.LeNet import LeNet
from modules.AL.models.FCN import FCN
from modules.AL.models.FDN import FDN
from modules.AL.models.LSTM import LSTM
from modules.AL.pt_utils import test_torch, test_torch_ensemble, train_test_nb_model
from tqdm import trange


def process(options):
    result_dir = '../../../results/'
    models_dir = '../../../models/'

    # Loop over N runs
    report_paths = {'softmax': [],
                    'weighted_softmax': [],
                    'naive_bayes': [],
                    'majority_vote': []}
    for i in range(options.runs):
        ######### Global Run Settings ###########
        np.random.seed(i)
        torch.manual_seed(i)
        random.seed(i)

        if options.verbose:
            print('Run %d / %d' % (i+1, options.runs))

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
        # Shapes
        print('TrainX:', trainX.shape)
        print('ValX:', valX.shape)
        print('TestX:', testX.shape)
        print('Classes:', n_classes)

        # Convert to channels first if not using LSTM
        trainX = torch.tensor(trainX.transpose(0,2,1)) if not options.architecture == 'LSTM' else torch.tensor(trainX)
        testX = torch.tensor(testX.transpose(0,2,1)) if not options.architecture == 'LSTM' else torch.tensor(testX)
        valX = torch.tensor(valX.transpose(0,2,1)) if not options.architecture == 'LSTM' else torch.tensor(valX)

        trainY = torch.tensor(trainY)
        testY = torch.tensor(testY)
        valY = torch.tensor(valY)

        # Split data
        federated_trainX, federated_trainY = make_federated_data(trainX, trainY, n_clients=options.n_clients, stratify=options.use_stratified)
        federated_valX, federated_valY = make_federated_data(valX, valY, n_clients=options.n_clients, stratify=options.use_stratified)

        lst_model_paths = []
        lst_model_val_acc = []

        for CLIENT_ID in range(options.n_clients):
            
            # Get corresponding data fold
            trainX = federated_trainX[CLIENT_ID]
            trainY = federated_trainY[CLIENT_ID]
            valX = federated_valX[CLIENT_ID]
            valY = federated_valY[CLIENT_ID]

            print('TrainX:', trainX.shape)

            sequence_length = trainX.shape[-1] if not options.architecture == 'LSTM' else trainX.shape[1]
            sequence_channels = trainX.shape[1] if not options.architecture == 'LSTM' else trainX.shape[-1]

            ##### Model Architecture ######
            architecture_func = {'AlexNet': AlexNet, 'LSTM': LSTM,
                             'FCN': FCN, 'FDN': FDN, 'LeNet': LeNet}

            model = architecture_func[options.architecture](in_width=sequence_length, in_channels=sequence_channels, num_classes=n_classes)

            ##### PyTorch Datasets & Dataloaders ######
            train_dataset = GenericDataset(x=trainX, y=trainY)
            val_dataset = GenericDataset(x=valX, y=valY)
            test_dataset = GenericDataset(x=testX, y=testY)

            train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                       batch_size=options.batch_size,
                                                       shuffle=True)
            val_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                     batch_size=options.batch_size)
            test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                      batch_size=options.batch_size)


            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.SGD(model.parameters(), lr=options.initial_lr)

            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            ##### Create Model and Results folder #####
            stratification = 'strat' if options.use_stratified else 'nonstrat'
            experiment_directory = 'BS-%d_LR-%.4f_strat-%s_NClients-%d' % (options.batch_size, options.initial_lr, stratification, options.n_clients)
            run_directory = 'run-%d'%i
            model_name = 'Client-%d.ckpt'%(CLIENT_ID)

            cur_result_dir = os.path.join(result_dir, 'FE_' + dataset_name, options.architecture, experiment_directory, run_directory)
            make_directory_if_not_exists(cur_result_dir)
            
            model_save_path = os.path.join(models_dir, 'FE_' + dataset_name, options.architecture, experiment_directory, run_directory)
            make_directory_if_not_exists(model_save_path)
            model_save_path = os.path.join(model_save_path, model_name)
            
            if os.path.isfile(model_save_path):
                model.load_state_dict(torch.load(model_save_path))
                print('Trained model found: ', model_save_path)

                val_acc =  test_torch(model=model, test_loader=val_loader, device=device, return_accuracy=True)
                lst_model_val_acc.append(val_acc)
            else:

                # Train the model
                print('\n--- Training Network ---')

                global_step = 0
                ES_counter = 0
                LR_counter = 0
                min_loss = sys.float_info.max
                current_lr = options.initial_lr
                current_val_loss = 0
                current_val_acc = 0
                current_train_acc = 0

                model.to(device)
                e = trange(options.epochs, desc='Bar desc', leave=True)
                for epoch in e:
                    model.train()
                    
                    correct = 0
                    total = 0
                    
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

                        # Compute accuracy
                        _, predicted = torch.max(outputs.data, 1)
                        total += y.size(0)
                        correct += (predicted == y).sum().cpu().data.numpy()

                        if (global_step+1) % 1 == 0:
                            e.set_description('Epoch [{}/{}], Step: {}, Loss: {:.4f}, Acc: {:.4f}, ESC: {}/{}, LR: {:.4f}, Val_Loss: {:.4f}, Val_Acc: {:.4f}'.format(epoch + 1, options.epochs, global_step + 1,  loss.item(), current_train_acc, ES_counter, options.es_patience, current_lr, current_val_loss, current_val_acc))
                            e.refresh()

                        global_step += 1
                    
                    current_train_acc = np.true_divide(correct, total)
                    e.set_description('Epoch [{}/{}], Step: {}, Loss: {:.4f}, Acc: {:.4f}, ESC: {}/{}, LR: {:.4f}, Val_Loss: {:.4f}, Val_Acc: {:.4f}'.format(epoch + 1, options.epochs, global_step + 1,  loss.item(), current_train_acc, ES_counter, options.es_patience, current_lr, current_val_loss, current_val_acc))
                    e.refresh()
                        
                    # Validate
                    correct = 0
                    total = 0
                    total_val_loss = 0

                    with torch.no_grad():
                        model.eval()
                        for step, (x, y) in enumerate(val_loader):
                                    
                            x = x.to(device, non_blocking=True)
                            y = y.to(device, non_blocking=True)

                            # Forward pass
                            outputs = model(x.float())
                            
                            # Compute accuracy
                            _, predicted = torch.max(outputs.data, 1)
                            total += y.size(0)
                            correct += (predicted == y).sum().cpu().data.numpy()

                            current_val_loss += criterion(outputs, y).item()
                            
                        current_val_loss = current_val_loss / total
                        current_val_acc = np.true_divide(correct, total)
                        e.set_description('Epoch [{}/{}], Step: {}, Loss: {:.4f}, Acc: {:.4f}, ESC: {}/{}, LR: {:.4f}, Val_Loss: {:.4f}, Val_Acc: {:.4f}'.format(epoch + 1, options.epochs, global_step + 1,  loss.item(), current_train_acc, ES_counter, options.es_patience, current_lr, current_val_loss, current_val_acc))
                        e.refresh()

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
                                
                                for g in optimizer.param_groups:
                                    g['lr'] = current_lr
                                
                                LR_counter = 0

                            ES_counter += 1
                            LR_counter += 1

                global_step = 0

                # Save Model
                torch.save(model.state_dict(), model_save_path)

                lst_model_val_acc.append(current_val_acc)

            # Test Model
            filename = 'Client_%d.txt'%CLIENT_ID
            res_save_path = os.path.join(cur_result_dir, filename)
            test_torch(model=model, 
                test_loader=test_loader, 
                device=device,
                save_path=res_save_path)

            lst_model_paths.append(model_save_path)

        ######### Load models and perform ensembling ###########
        print(lst_model_val_acc)

        models = []
        for model_path in lst_model_paths:
            models.append(architecture_func[options.architecture](in_width=sequence_length, in_channels=sequence_channels, num_classes=n_classes))
            models[-1].load_state_dict(torch.load(model_path))
        
        model_weights = [i/sum(lst_model_val_acc) for i in lst_model_val_acc]
        print('Model Softmax Weights: ', model_weights)
        
        res_save_path = os.path.join(cur_result_dir, 'Ensemble_Softmax_Averaging.txt')
        test_torch_ensemble(models=models, test_loader=test_loader, device=device, ensemble_method='average_output', ensemble_weights=None, save_path=res_save_path)
        report_paths['softmax'].append(res_save_path.replace('.txt', '.pickle'))

        res_save_path = os.path.join(cur_result_dir, 'Ensemble_Weighted_Softmax_Averaging.txt')
        test_torch_ensemble(models=models, test_loader=test_loader, device=device, ensemble_method='average_output', ensemble_weights=model_weights, save_path=res_save_path)
        report_paths['weighted_softmax'].append(res_save_path.replace('.txt', '.pickle'))

        res_save_path = os.path.join(cur_result_dir, 'Ensemble_Naive_Bayes.txt')
        train_test_nb_model(models=models, train_loader=train_loader, test_loader=test_loader, device=device, save_path=res_save_path)
        report_paths['naive_bayes'].append(res_save_path.replace('.txt', '.pickle'))

        res_save_path = os.path.join(cur_result_dir, 'Ensemble_Majority_Vote.txt')
        test_torch_ensemble(models=models, test_loader=test_loader, device=device, ensemble_method='majority_vote', save_path=res_save_path)
        report_paths['majority_vote'].append(res_save_path.replace('.txt', '.pickle'))

    ###### Create mean eval reports #########
    if options.save_mcr:

        # Loop over different ensembling approaches
        for ens_approach in report_paths.keys():
            mean_report_path = os.path.join(result_dir, 'FE_' + dataset_name, options.architecture, experiment_directory, 'mean-report_' + ens_approach + '.txt')
            print(ens_approach)
            print(report_paths[ens_approach])
            mean_cr_utils.compute_meanclassification_report(
                report_paths[ens_approach], save=mean_report_path, verbose=options.verbose, store_dict=True)
    

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
    parser.add_option("--num_cpus", action="store", type=int,
                      dest="num_cpus", default=11, help="Number of cpus to use for preprocessing")

    ####### Perform baseline model #########
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
