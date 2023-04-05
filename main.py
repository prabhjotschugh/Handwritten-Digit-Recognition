import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Loading the Dataset
mnist = tf.keras.datasets.mnist
 
# Split the dataset for training and testing
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the data (converting every value between 0 and 1)
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# Working on the Neural Network (sequential is the basic Neural Network)
model = tf.keras.Sequential()

#adding layers
# Add one flattened input layer for the pixels
model.add(tf.keras.layers.Flatten(input_shape=(28,28)))

# Add two dense hidden layers
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(128, activation='relu'))

# Add one dense output layer for the 10 digits
model.add(tf.keras.layers.Dense(10, activation='softmax'))

# Compiling and optimizing the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Training the model
model.fit(x_train, y_train, epochs=10)

#Saving the model
model.save('handwritten.model')

model = tf.keras.models.load_model('handwritten.model')

# Evaluating the model
loss, accuracy = model.evaluate(x_test, y_test)

print (loss)
print(accuracy*100)


        