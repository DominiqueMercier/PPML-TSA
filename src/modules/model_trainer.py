import os

import tensorflow as tf


def train(model, trainX, trainY, validation_data=None, epochs=100, batch_size=32, model_path=None, verbose=0):
    save_path = model_path
    if save_path is None:
        save_path = '../tmp_model.h5'
    earlyStop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', min_delta=0, patience=10, verbose=verbose, mode='auto')
    lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                                      patience=4, verbose=verbose, mode='auto', cooldown=0, min_lr=0)
    mcp_save = tf.keras.callbacks.ModelCheckpoint(
        save_path, save_best_only=False, save_weights_only=True, monitor='val_loss')
    callback_list = [earlyStop, lr_reducer, mcp_save]

    if validation_data is None:
        model.fit(trainX, trainY, validation_split=0.3, use_multiprocessing=True,
                  epochs=epochs, batch_size=batch_size, verbose=verbose, callbacks=callback_list)
    else:
        model.fit(trainX, trainY, validation_data=validation_data, use_multiprocessing=True,
                  epochs=epochs, batch_size=batch_size, verbose=verbose, callbacks=callback_list)
    model.load_weights(save_path)
    if model_path is None:
        os.remove(save_path)
    else:
        model.save(save_path)
    if verbose:
        if not model_path is None:
            print('Model location:', model_path)