#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import LSTM, Embedding, Dense
from sklearn.metrics import classification_report


class SentimentAnalysis:
    def __init__(self, load_model=False):
        # ------------------------------------------------ #
        #   Modules
        # ------------------------------------------------ #
        self._folder = "../dataset/"
        self._train_data = None
        self._test_data = None
        self._Y = None
        self._train = None
        self._test = None
        self._total_data = None
        self._keras_tokenizer = Tokenizer()
        self._sequence_data = None
        self._X = None
        self._X_test = None
        self._Y_test = None
        self._X_train = None
        self._X_valid = None
        self._Y_train = None
        self._Y_valid = None
        self._model = None
        self._log = None

        self._read_data()
        self._data_preparation()
        self._pre_processing()
        self._data_segmentation()
        if load_model is False:
            self._lstm()
            self._training()
            self._result_visualization()
        else:
            self._load_model()

    # ------------------------------------------------ #
    #   Read dataset
    # ------------------------------------------------ #
    def _read_data(self):
        self._train_data = pd.read_csv(self._folder + 'train.tsv', delimiter='\t')
        self._test_data = pd.read_csv(self._folder + 'test.tsv', delimiter='\t')
        print("The length of train dataset is " + str(len(self._train_data)))
        print("The length of train dataset is " + str(len(self._test_data)))

    # ------------------------------------------------ #
    #   Data Preparation
    # ------------------------------------------------ #
    def _data_preparation(self):
        if self._train_data is None or self._test_data is None:
            return
        self._Y = self._train_data['Sentiment']
        self._Y = np.array(self._Y)
        self._Y = self._Y.reshape(-1, 1)

        # Dummy variables
        self._Y = to_categorical(self._Y)
        print("The shape of Y is " + str(self._Y.shape))

        # Training data and Test data
        self._train = self._train_data['Phrase']
        self._test = self._test_data['Phrase']

        # Combination of these two data
        self._total_data = pd.concat([self._train, self._test], axis=0)
        print("The length of total data is " + str(len(self._total_data)))

    # ------------------------------------------------ #
    #   Preprocessing
    #   1. Text Segmentation
    #   2. Removing unnecessary data ('!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~')
    #   3. Tokenizer
    # ------------------------------------------------ #
    def _pre_processing(self):
        if self._total_data is None:
            return
        total_np = np.array(self._total_data)
        total_list = total_np.tolist()

        # Text segmentation and Removing unnecessary data
        self._keras_tokenizer.fit_on_texts(total_list)
        print("The number of learned words is " + str(len(self._keras_tokenizer.word_index)))

        # Tokenizer
        self._sequence_data = self._keras_tokenizer.texts_to_sequences(total_list)
        self._X = keras.preprocessing.sequence.pad_sequences(self._sequence_data, padding="post")

    # ------------------------------------------------ #
    #   Data Segmentation
    # ------------------------------------------------ #
    def _data_segmentation(self):
        train = self._X[:156060]
        test = self._X[156060:]

        # The shape of test is one-hot
        X_train, self._X_test, Y_train, self._Y_test = train_test_split(train, self._Y, test_size=0.2, random_state=0)
        self._X_train, self._X_valid, self._Y_train, self._Y_valid = train_test_split(X_train, Y_train, test_size=0.2,
                                                                                      random_state=0)

    # ------------------------------------------------ #
    #   LSTM model
    # ------------------------------------------------ #
    def _lstm(self):
        # Embedding layer
        self._model = keras.Sequential()
        self._model.add(Embedding(17781, 64, mask_zero=True))
        # LSTM layer
        self._model.add(LSTM(64, return_sequences=True))
        self._model.add(LSTM(32))
        self._model.add(Dense(5, activation='sigmoid'))

        self._model.summary()

    # ------------------------------------------------ #
    #   Training
    # ------------------------------------------------ #
    def _training(self):
        if self._model is None:
            return
        self._model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
        self._log = self._model.fit(self._X_train, self._Y_train, epochs=100, batch_size=2048,
                              callbacks=[keras.callbacks.EarlyStopping(monitor='val_loss',
                                                                       min_delta=0,
                                                                       patience=20,
                                                                       verbose=1)],
                              validation_data=(self._X_valid, self._Y_valid))

        Y_test_ = np.argmax(self._Y_test, axis=1)
        Y_pred = self._model.predict_classes(self._X_test)
        print(classification_report(Y_test_, Y_pred))
        self._model.save("../model/")

    # ------------------------------------------------ #
    #   Result Visualization
    # ------------------------------------------------ #
    def _result_visualization(self):
        plt.plot(self._log.history['loss'], label='loss')
        plt.plot(self._log.history['val_loss'], label='val_loss')
        plt.legend(frameon=False)
        plt.xlabel("epochs")
        plt.ylabel("crossentropy")
        plt.show()

        plt.plot(self._log.history['accuracy'], label='accuracy')
        plt.plot(self._log.history['val_accuracy'], label='val_accuracy')
        plt.legend(frameon=False)
        plt.xlabel("epochs")
        plt.ylabel("crossentropy")
        plt.show()

    # ------------------------------------------------ #
    #   Loading model
    # ------------------------------------------------ #
    def _load_model(self):
        self._model = keras.models.load_model('../model/')
        Y_test_ = np.argmax(self._Y_test, axis=1)
        Y_pred = self._model.predict_classes(self._X_test)
        print(classification_report(Y_test_, Y_pred))


if __name__ == "__main__":
    sen = SentimentAnalysis(load_model=True)
