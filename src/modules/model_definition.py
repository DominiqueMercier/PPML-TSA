import tensorflow as tf


class AlexNet(tf.keras.Sequential):
    def __init__(self):
        super(AlexNet, self).__init__()

    def build_1d(self, input_shape, n_classes, use_batch_shape=False, activation='softmax', batch_norm=True, verbose=0):
        model = tf.keras.models.Sequential()
        if use_batch_shape:
            model.add(tf.keras.layers.Conv1D(filters=96, kernel_size=11, strides=4, padding="same", activation='relu',
                                             batch_input_shape=input_shape))
        else:
            model.add(tf.keras.layers.Conv1D(filters=96, kernel_size=11, strides=4, padding="same", activation='relu',
                                             input_shape=input_shape))

        if batch_norm:
            model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.MaxPool1D(
            pool_size=3, strides=2, padding="same"))
        model.add(tf.keras.layers.Conv1D(filters=256, kernel_size=5,
                                         strides=1, activation='relu', padding="same"))
        if batch_norm:
            model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.MaxPool1D(
            pool_size=3, strides=2, padding="same"))
        model.add(tf.keras.layers.Conv1D(filters=384, kernel_size=3,
                                         strides=1, activation='relu', padding="same"))
        if batch_norm:
            model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Conv1D(filters=384, kernel_size=1,
                                         strides=1, activation='relu', padding="same"))
        if batch_norm:
            model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Conv1D(filters=256, kernel_size=1,
                                         strides=1, activation='relu', padding="same"))
        if batch_norm:
            model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.MaxPool1D(
            pool_size=3, strides=2, padding="same"))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(4096, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.Dense(4096, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.Dense(n_classes, activation=activation))

        if verbose:
            model.summary()
        return model

    def build_2d(self, input_shape, n_classes, use_batch_shape=False, activation='softmax', batch_norm=True, verbose=0):
        model = tf.keras.models.Sequential()
        if use_batch_shape:
            model.add(tf.keras.layers.Conv2D(filters=96, kernel_size=(11, 11), strides=(
                4, 4), padding="same", activation='relu', batch_input_shape=input_shape))
        else:
            model.add(tf.keras.layers.Conv2D(filters=96, kernel_size=(11, 11), strides=(
                4, 4), padding="same", activation='relu', input_shape=input_shape))

        if batch_norm:
            model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.MaxPool2D(pool_size=(
            3, 3), strides=(2, 2), padding="same"))
        model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(
            5, 5), strides=(1, 1), activation='relu', padding="same"))
        if batch_norm:
            model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.MaxPool2D(pool_size=(
            3, 3), strides=(2, 2), padding="same"))
        model.add(tf.keras.layers.Conv2D(filters=384, kernel_size=(
            3, 3), strides=(1, 1), activation='relu', padding="same"))
        if batch_norm:
            model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Conv2D(filters=384, kernel_size=(
            1, 1), strides=(1, 1), activation='relu', padding="same"))
        if batch_norm:
            model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(
            1, 1), strides=(1, 1), activation='relu', padding="same"))
        if batch_norm:
            model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.MaxPool2D(pool_size=(
            3, 3), strides=(2, 2), padding="same"))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(4096, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.Dense(4096, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.Dense(n_classes, activation=activation))

        if verbose:
            model.summary()
        return model


class LSTM(tf.keras.Sequential):
    def __init__(self):
        super(LSTM, self).__init__()

    def build_default(self, input_shape, n_classes, use_batch_shape=False, activation='softmax', verbose=0):
        model = tf.keras.models.Sequential()
        if use_batch_shape:
            model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(512, return_sequences=True),
                                                    batch_input_shape=input_shape))
        else:
            model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(512, return_sequences=True),
                                                    input_shape=input_shape))

        model.add(tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(512)))
        model.add(tf.keras.layers.Dense(n_classes, activation=activation))

        if verbose:
            model.summary()
        return model


class FDN(tf.keras.Sequential):
    def __init__(self):
        super(FDN, self).__init__()

    def build_default(self, input_shape, n_classes, use_batch_shape=False, activation='softmax', verbose=0):
        model = tf.keras.models.Sequential()
        if use_batch_shape:
            model.add(tf.keras.layers.Flatten(batch_input_shape=input_shape))
        else:
            model.add(tf.keras.layers.Flatten(input_shape=input_shape))

        model.add(tf.keras.layers.Dense(4096, activation='relu'))
        model.add(tf.keras.layers.Dense(4096, activation='relu'))
        model.add(tf.keras.layers.Dense(n_classes, activation=activation))

        if verbose:
            model.summary()
        return model


class FCN(tf.keras.Sequential):
    def __init__(self):
        super(FCN, self).__init__()

    def build_default(self, input_shape, n_classes, use_batch_shape=False, activation='softmax', verbose=0):
        model = tf.keras.models.Sequential()
        if use_batch_shape:
            model.add(tf.keras.layers.Conv1D(96, 11, strides=4, padding="same",
                      activation='relu', batch_input_shape=input_shape))
        else:
            model.add(tf.keras.layers.Conv1D(
                96, 11, strides=4, padding="same", activation='relu', input_shape=input_shape))

        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Conv1D(
            256, 5, strides=2, padding="same", activation='relu'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Conv1D(
            384, 3, strides=1, padding="same", activation='relu'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Conv1D(
            384, 1, strides=1, padding="same", activation='relu'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Conv1D(
            256, 1, strides=1, padding="same", activation='relu'))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Conv1D(n_classes, 1, activation=activation))
        model.add(tf.keras.layers.GlobalMaxPool1D())

        if verbose:
            model.summary()
        return model


class LeNet(tf.keras.Sequential):
    def __init__(self):
        super(LeNet, self).__init__()

    def build_default(self, input_shape, n_classes, use_batch_shape=False, activation='softmax', verbose=0):
        model = tf.keras.models.Sequential()
        if use_batch_shape:
            model.add(tf.keras.layers.Conv1D(
                6, 5, activation='tanh', batch_input_shape=input_shape))
        else:
            model.add(tf.keras.layers.Conv1D(
                6, 5, activation='tanh', input_shape=input_shape))

        model.add(tf.keras.layers.Activation('sigmoid'))
        model.add(tf.keras.layers.AveragePooling1D(2))
        model.add(tf.keras.layers.Conv1D(16, 5, activation='tanh'))
        model.add(tf.keras.layers.Activation('sigmoid'))
        model.add(tf.keras.layers.AveragePooling1D(2))
        model.add(tf.keras.layers.Conv1D(120, 5, activation='tanh'))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(84, activation='tanh'))
        model.add(tf.keras.layers.Dense(n_classes, activation=activation))

        if verbose:
            model.summary()
        return model
