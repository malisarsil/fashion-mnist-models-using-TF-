# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 21:05:34 2021

@author: sarsi
"""

#First model
import numpy as np
import tensorflow as tf
import keras
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

training_images = train_images/255.0
test_images = test_images/255.0


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs = {}):
        if (logs.get('acc') > 0.6):
            print("\n accuracy is as wanted. So cancelling training.")
            self.model.stop_training = True

model = keras.Sequential([keras.layers.Flatten(input_shape = (28,28)),
                          keras.layers.Dense(128, activation='relu'), 
                          keras.layers.Dense(10, activation="softmax")])
model.compile(optimizer = tf.keras.optimizers.Adam(), loss = "sparse_categorical_crossentropy", metrics=["accuracy"])

            
callbacks = myCallback()

model.fit(train_images, train_labels, epochs = 10, callbacks = [callbacks])
model.evaluate(test_images, test_labels)
