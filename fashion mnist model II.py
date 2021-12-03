
# 2. model
import numpy as np
import tensorflow as tf
import keras
import matplotlib.pyplot as plt

class mycallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if logs.get('loss') is not None and logs.get('loss') < 0.4:
      print("\nReached 99% accuracy so cancelling training!")
      self.model.stop_training = True
      
callback = mycallback()
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

train_images = train_images/255.0
test_images = test_images/255.0

model = keras.Sequential([tf.keras.layers.Conv2D(64, (3,3), 
activation = "relu", input_shape = (28,28,1))
,tf.keras.layers.MaxPooling2D(2,2), tf.keras.layers.Conv2D(64, (3,3), 
activation = "relu"),tf.keras.layers.MaxPooling2D(2,2)
,keras.layers.Flatten(input_shape = (28,28)),
keras.layers.Dense(128, activation=tf.nn.relu), 
keras.layers.Dense(10, activation=tf.nn.softmax)])


model.compile(optimizer=tf.keras.optimizers.Adam(), loss = "sparse_categorical_crossentropy")
model.fit(train_images, train_labels, epochs = 5, callbacks=[callback])
results = model.evaluate(test_images, test_labels)
