import os

import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
from modules import utils
from modules.DM import adaptive_fed_avg, callbacks


def make_federated_data(data, labels, n_clients=2, stratify=True, return_state=False, random_state=0):
    split_size = 1 / n_clients
    divisor = 1
    restX, restY = data, labels
    federated_data = []
    if stratify:
        state = True
    for _ in range(n_clients-1):
        if stratify:
            restX, restY, tmpX, tmpY, tmp_state = utils.perform_datasplit(
                restX, restY, test_split=split_size / divisor, stratify=True, return_state=True, random_state=random_state)
            state = min(state, tmp_state)
        else:
            restX, restY, tmpX, tmpY = utils.perform_datasplit(
                restX, restY, test_split=split_size / divisor, stratify=False, random_state=random_state)
            state = False
        divisor -= split_size
        d = tf.data.Dataset.from_tensor_slices((tmpX, tmpY))
        federated_data.append(d)
    d = tf.data.Dataset.from_tensor_slices((restX, restY))
    federated_data.append(d)
    if return_state:
        return federated_data, state
    return federated_data


def preporcess_federated_data(fed_data, num_epochs=1, batch_size=32):
    new_data = []
    for i in range(len(fed_data)):
        new_data.append(fed_data[i].repeat(num_epochs).shuffle(batch_size).batch(batch_size))
    return new_data

def create_federated_processes(federated_dict, server_learning_rate, client_learning_rate=None, adaptive=False, stepwise=False):
    def model_fn():
        if federated_dict['architecture'] == 'AlexNet':
            model = federated_dict['architecture_func'][federated_dict['architecture']](federated_dict['input_shape'], federated_dict['n_classes'], activation='softmax', batch_norm=False, verbose=0)
        else:
            model = federated_dict['architecture_func'][federated_dict['architecture']](federated_dict['input_shape'], federated_dict['n_classes'], activation='softmax', verbose=0)
        return tff.learning.from_keras_model(model,
            input_spec=federated_dict['input_spec'],
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

    if client_learning_rate is None and not stepwise:
        client_learning_rate = server_learning_rate
    # compile
    if not adaptive:
        if not stepwise:
            federated_train_process = tff.learning.build_federated_averaging_process(model_fn,
                client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=client_learning_rate),
                server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=server_learning_rate),
                model_update_aggregation_factory=tff.learning.robust_aggregator(zeroing = True,clipping = True))
        else:
            federated_train_process = tff.learning.build_federated_sgd_process(model_fn,
                server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=server_learning_rate),
                model_update_aggregation_factory=tff.learning.robust_aggregator(zeroing = True,clipping = True))
    else:
        #adaptive compile
        client_lr_callback = callbacks.create_reduce_lr_on_plateau(learning_rate=client_learning_rate, patience=np.iinfo(np.int32).max)
        server_lr_callback = callbacks.create_reduce_lr_on_plateau(learning_rate=server_learning_rate, patience=np.iinfo(np.int32).max)
        federated_train_process = adaptive_fed_avg.build_fed_avg_process(model_fn, client_lr_callback, server_lr_callback,
            client_optimizer_fn=tf.keras.optimizers.SGD, server_optimizer_fn=tf.keras.optimizers.SGD)

    federated_eval_process = tff.learning.build_federated_evaluation(model_fn)
    return federated_train_process, federated_eval_process

def traintf(state, federated_dict, federated_train_process, federated_train_pre, federated_eval_process=None, federated_val=None, 
            n_clients=None, epochs=100, batch_size=32, lr_decay=False, model_path=None, verbose=0):
    if federated_dict['architecture'] == 'AlexNet':
        model_extern = federated_dict['architecture_func'][federated_dict['architecture']](
            federated_dict['input_shape'], federated_dict['n_classes'], activation='softmax', batch_norm=False, verbose=0)
    else:
        model_extern = federated_dict['architecture_func'][federated_dict['architecture']](
            federated_dict['input_shape'], federated_dict['n_classes'], activation='softmax', verbose=0)
    save_path = model_path
    
    best_loss, earlyc, lrc= np.sys.float_info.max, 0, 0
    for i in range(epochs):
        print('Epoch %i / %i' % (i+1, epochs))
        participating_clients = np.arange(len(federated_train_pre)) 
        if not n_clients is None:
            participating_clients = np.random.permutation(participating_clients)[:n_clients]
        # Run a training pass
        training_sets = preporcess_federated_data(
            [federated_train_pre[i] for i in participating_clients], batch_size=32)
        state, metrics = federated_train_process.next(state, training_sets)
        if not lr_decay:
            loss, acc = metrics['train']['loss'], metrics['train']['sparse_categorical_accuracy']
        else:
            loss, acc = metrics['during_training']['loss'], metrics['during_training']['sparse_categorical_accuracy']
        print('client_loss:\t%.4f |\tclient_acc:\t%.4f' % (loss, acc))
        if not federated_val is None:
            train_metrics = federated_eval_process(state.model, training_sets)
            train_loss, train_acc = train_metrics['loss'], train_metrics['sparse_categorical_accuracy']
            print('train_loss:\t%.4f |\ttrain_acc:\t%.4f' % (train_loss, train_acc))
            # Run an evaluation pass
            eval_sets = [federated_val[i] for i in participating_clients]
            val_metrics = federated_eval_process(state.model, eval_sets)
            val_loss, val_acc = val_metrics['loss'], val_metrics['sparse_categorical_accuracy']
            print('val_loss:\t%.4f |\tval_acc:\t%.4f' % (val_loss, val_acc))
        else:
            val_loss = loss
        if val_loss < best_loss:
            best_loss = val_loss
            earlyc = 0
            lrc = 0
            if not lr_decay:
                state.model.assign_weights_to(model_extern)
            else:
                for src, tar in zip(state.model.trainable, model_extern.trainable_weights):
                    tar.assign(src)
                for src, tar in zip(state.model.non_trainable, model_extern.non_trainable_weights):
                    tar.assign(src)
        else:
            lrc += 1
            earlyc += 1
        if lr_decay and lrc == 4:
            lrc = 0
            state.client_lr_callback.learning_rate = state.client_lr_callback.learning_rate * 0.5
            state.server_lr_callback.learning_rate = state.server_lr_callback.learning_rate * 0.5
            print('Reduced learning rate to:', state.client_lr_callback.learning_rate)
            print('Reduced learning rate to:', state.server_lr_callback.learning_rate)
        if earlyc == 10:
            print('Early Stopping')
            break
    if not model_path is None:
        model_extern.save(save_path)
    if verbose:
        if not model_path is None:
            print('Model location:', model_path)
    return model_extern
