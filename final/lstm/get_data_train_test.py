# data augmentation and preprocessing
from load_data import load_base_data
import random
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
import numpy as np


def get_data():
    """
    To get X_train, Y_train, X_test, Y_test

    this function is to get the data as tensors after preprocessing and augmentation

    Parameters:
        None

    Returns:
        X_train (int): training data
        Y_train (int): training labels
        X_test (int): test data
        Y_test (int): test labels
    """
    x, y = load_base_data()

    # train test split
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

    x_train = tf.convert_to_tensor(x_train)
    x_test = tf.convert_to_tensor(x_test)
    y_train = tf.convert_to_tensor(y_train)
    y_test = tf.convert_to_tensor(y_test)

    # data augmentation
    X_train_new = x_train
    X_train_new = tf.concat([X_train_new, tf.map_fn(sub_rand, x_train)], 0)
    X_train_new = tf.concat([X_train_new, tf.map_fn(add_rand, x_train)], 0)
    y_train_new = y_train
    y_train_new = tf.concat([y_train_new, y_train], 0)
    y_train_new = tf.concat([y_train_new, y_train], 0)

    # preprocessing
    X_train, Y_train, X_test, Y_test = preprocess(X_train_new, y_train_new, x_test, y_test)

    return X_train, Y_train, X_test, Y_test


def preprocess(X_train, Y_train, X_test, Y_test):
    """
    preprocess data

    normalize training and test data tensors
    convert training and test label tensors to one-hot vectors

    Parameters:
        X_train (int): training data
        Y_train (int): training labels
        X_test (int): test data
        Y_test (int): test labels

    Returns:
        X_train (int): training data
        Y_train (int): training labels
        X_test (int): test data
        Y_test (int): test labels
    """
    # encode the output
    encoder = LabelEncoder()
    encoder.fit(Y_train)
    Y_train = encoder.transform(Y_train)
    Y_train = np_utils.to_categorical(Y_train)

    encoder = LabelEncoder()
    encoder.fit(Y_test)
    Y_test = encoder.transform(Y_test)
    Y_test = np_utils.to_categorical(Y_test)

    X_train = X_train / 360
    X_test = X_test / 360

    return X_train, Y_train, X_test, Y_test


def add_rand(angle):
    angle = angle + random.randint(0, 5)
    return angle


def sub_rand(angle):
    angle = angle - random.randint(0, 5)
    return abs(angle)


def get_data_numpy():
    """
    To get X_train, Y_train, X_test, Y_test

    this function is to get the data as numpy arrays after preprocessing and augmentation

    Parameters:
        None

    Returns:
        X_train (int): training data
        Y_train (int): training labels
        X_test (int): test data
        Y_test (int): test labels
    """


    x, y = load_base_data()

    # train test split
    from sklearn.model_selection import train_test_split

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

    # data augmentation
    X_train_new = x_train
    sum = lambda x: x + random.randint(0, 5)
    sub = lambda x: abs(x - random.randint(0, 5))
    X_train_new = np.concatenate((X_train_new, np.vectorize(sum)(x_train)), axis=0)
    X_train_new = np.concatenate((X_train_new, np.vectorize(sub)(x_train)), axis=0)
    y_train_new = y_train
    y_train_new = np.concatenate((y_train_new, y_train), axis=0)
    y_train_new = np.concatenate((y_train_new, y_train), axis=0)

    # preprocessing
    X_train, Y_train, X_test, Y_test = preprocess_numpy(X_train_new, y_train_new, x_test, y_test)

    return X_train, Y_train, X_test, Y_test


def preprocess_numpy(X_train, Y_train, X_test, Y_test):
    """
        preprocess data

        normalize training and test data array
        convert training and test label array to one-hot vectors

        Parameters:
            X_train (int): training data
            Y_train (int): training labels
            X_test (int): test data
            Y_test (int): test labels

        Returns:
            X_train (int): training data
            Y_train (int): training labels
            X_test (int): test data
            Y_test (int): test labels
        """
    # encode the output
    encoder = LabelEncoder()
    encoder.fit(Y_train)
    Y_train = encoder.transform(Y_train)
    Y_train = np_utils.to_categorical(Y_train)

    encoder = LabelEncoder()
    encoder.fit(Y_test)
    Y_test = encoder.transform(Y_test)
    Y_test = np_utils.to_categorical(Y_test)

    X_train = np.divide(X_train, 360)
    X_test = np.divide(X_test, 360)

    return X_train, Y_train, X_test, Y_test
