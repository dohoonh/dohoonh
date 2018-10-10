import os
import sys
import glob
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from keras.utils import np_utils
from keras.layers import (Conv1D, MaxPool1D, BatchNormalization,
                          Dense, Dropout, Activation, Flatten, Reshape, Input)
from keras.optimizers import SGD
from keras.models import Model


NB_EPOCHS = 100
BAT_SIZE = 64
train_data = 'D:\dohoonh_dataset/_3000train_600test/csv_static_inspi_spark_hexa/train_data.csv'
test_data = 'D:\dohoonh_dataset/_3000train_600test/csv_static_inspi_spark_hexa/test_data.csv'


def sample_cnn_generator(input_dim, n_outputs=3, activation='relu',
                         kernel_initializer='he_uniform', dropout_rate=0.5):

    input_data = Input(shape=(input_dim, 1))
    # 1024 X 1
    net = Conv1D(128, 3, strides=1, padding='same',
                 kernel_initializer=kernel_initializer)(input_data)
    net = BatchNormalization()(net)
    net = Activation(activation)(net)
    # 1024 X 128
    net = Conv1D(128, 3, padding='same',
                 kernel_initializer=kernel_initializer)(net)
    net = BatchNormalization()(net)
    net = Activation(activation)(net)
    net = MaxPool1D(4)(net)
    # 512 X 128
    net = Conv1D(128, 3, strides=1, padding='same',
                 kernel_initializer=kernel_initializer)(net)
    net = BatchNormalization()(net)
    net = Activation(activation)(net)
    net = MaxPool1D(4)(net)
    # 256 X 128
    net = Conv1D(256, 3, padding='same',
                 kernel_initializer=kernel_initializer)(net)
    net = BatchNormalization()(net)
    net = Activation(activation)(net)
    net = MaxPool1D(2)(net)
    # # 128 X 256
    # net = Conv1D(256, 3, padding='same',
    #              kernel_initializer=kernel_initializer)(net)
    # net = BatchNormalization()(net)
    # net = Activation(activation)(net)
    # net = MaxPool1D(2)(net)
    # # 64 X 256
    # net = Conv1D(256, 3, padding='same',
    #              kernel_initializer=kernel_initializer)(net)
    # net = BatchNormalization()(net)
    # net = Activation(activation)(net)
    # net = MaxPool1D(2)(net)
    # # 32 X 256
    # net = Conv1D(256, 3, padding='same',
    #              kernel_initializer=kernel_initializer)(net)
    # net = BatchNormalization()(net)
    # net = Activation(activation)(net)
    # net = MaxPool1D(2)(net)
    # # 16 X 256
    # net = Conv1D(256, 3, padding='same',
    #              kernel_initializer=kernel_initializer)(net)
    # net = BatchNormalization()(net)
    # net = Activation(activation)(net)
    # net = MaxPool1D(2)(net)
    # # 8 X 256
    # net = Conv1D(256, 3, padding='same',
    #              kernel_initializer=kernel_initializer)(net)
    # net = BatchNormalization()(net)
    # net = Activation(activation)(net)
    # net = MaxPool1D(2)(net)
    # # 4 X 256
    # net = Conv1D(512, 3, padding='same',
    #              kernel_initializer=kernel_initializer)(net)
    # net = BatchNormalization()(net)
    # net = Activation(activation)(net)
    # net = MaxPool1D(2)(net)
    # # 2 X 512
    # net = Conv1D(512, 3, padding='same',
    #              kernel_initializer=kernel_initializer)(net)
    # net = BatchNormalization()(net)
    # net = Activation(activation)(net)
    # net = MaxPool1D(2)(net)
    # # 1 X 512
    net = Conv1D(512, 1, padding='same',
                 kernel_initializer=kernel_initializer)(net)
    net = BatchNormalization()(net)
    net = Activation(activation)(net)
    # 1 X 512
    net = Dropout(dropout_rate)(net)
    net = Flatten()(net)

    predictions = Dense(units=n_outputs, activation='softmax')(net)

    model = Model(input=input_data, output=predictions)
    return model


def train():
    train_label = pd.read_csv(train_data)
    test_label = pd.read_csv(test_data)

    train_labels = train_label.ix[:, 0].values.astype('int32')
    x_train_val = train_label.ix[:, 1:].values.astype('str')
    x_train_complex = np.char.replace(x_train_val, 'i', 'j').astype(np.complex)
    x_train = np.real(x_train_complex)

    test_labels = test_label.ix[:, 0].values.astype('int32')
    x_test_val = test_label.ix[:, 1:].values.astype('str')
    x_test_complex = np.char.replace(x_test_val, 'i', 'j').astype(np.complex)
    x_test = np.real(x_test_complex)

    # convert list of labels to binary class matrix
    y_train = np_utils.to_categorical(train_labels)
    y_test = np_utils.to_categorical(test_labels)

    # pre-processing: divide by max and substract mean
    # scale = np.max(x_train, 1)
    # x_train /= scale
    # x_test /= scale

    # mean = np.std(x_train, 1)
    # x_train -= mean
    # x_test -= mean

    input_dim = x_train.shape[1]
    nb_classes = y_train.shape[1]

    x_train = np.expand_dims(x_train, axis=2)
    x_test = np.expand_dims(x_test, axis=2)
    # setup model
    model = sample_cnn_generator(
        input_dim,
        n_outputs=nb_classes,
        activation='relu',
        kernel_initializer='he_uniform',
        dropout_rate=0.5)

    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
    print("-- Training --")
    history_ft = model.fit(x_train, y_train, batch_size=BAT_SIZE, epochs=NB_EPOCHS, validation_split=0.05)
    # history_ft = model.fit(x_train, y_train, batch_size=BAT_SIZE, epochs=NB_EPOCHS, validation_data=(x_test, y_test))
    print("-- Evaluate --")
    scores = model.evaluate(x_test, y_test, batch_size=BAT_SIZE, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])
    model.summary()
    # model.save(args.output_model_file)

    plot_training(history_ft)


def plot_training(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))

    plt.plot(epochs, acc, 'r.')
    plt.plot(epochs, val_acc, 'r')
    plt.title('Training and validation accuracy')

    plt.figure()
    plt.plot(epochs, loss, 'r.')
    plt.plot(epochs, val_loss, 'r-')
    plt.title('Training and validation loss')
    plt.show()


if __name__ == "__main__":
    train()
