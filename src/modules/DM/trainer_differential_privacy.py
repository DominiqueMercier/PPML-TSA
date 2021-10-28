import os

import numpy as np


def traintf(sess, model, trainX, trainY, validation_data=None, epochs=100, batch_size=32, model_path=None, verbose=0):
    save_path = model_path
    if save_path is None:
        save_path = '../tmp_model.h5'

    best_loss, lrc, earlyc = np.sys.float_info.max, 0, 0
    for i in range(epochs):
        print('Epoch %i / %i' % (i+1, epochs))
        if validation_data is None:
            h = model.fit(trainX, trainY, validation_split=0.3, use_multiprocessing=True,
                          epochs=1, batch_size=batch_size, verbose=verbose)
        else:
            h = model.fit(trainX, trainY, validation_data=validation_data, use_multiprocessing=True,
                          epochs=1, batch_size=batch_size, verbose=verbose)
        val_loss = np.mean(
            h.history['val_loss'] if not validation_data is None else h.history['loss'])
        if val_loss < best_loss:
            best_loss = val_loss
            earlyc = 0
            lrc = 0
            model.save(save_path)
        else:
            earlyc += 1
            lrc += 1
        if lrc == 4:
            new_lr = sess.run(model.optimizer.lr.assign(
                model.optimizer.lr.read_value() * 0.5))
            lrc = 0
            print('Reduced learning rate to:', new_lr)
        if earlyc == 10:
            print('Early Stopping')
            break
    model.load_weights(save_path)
    if model_path is None:
        os.remove(save_path)
    else:
        model.save(save_path)
    if verbose:
        if not model_path is None:
            print('Model location:', model_path)
