import matplotlib.pyplot as plt
import tensorflow as tf
import os
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Reshaping the array to 4-dims so that it can work with the Keras API
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)
# Making sure that the values are float so that we can get decimal points after division
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# Normalizing the RGB codes by dividing it to the max RGB value.
x_train /= 255
x_test /= 255

path_save_model = 'save_weight_classifier/version_1.h5'

model1 = tf.keras.models.load_model(path_save_model)
model1.evaluate(x_test,y_test)
image_index = 4444
pred1 = model1.predict(x_test[image_index].reshape(1, 28, 28, 1))
# pred1: [[2.5758001e-13 1.5505588e-19 3.5310039e-13 1.6146470e-11 2.7523498e-07 2.8328062e-10 2.6123212e-15 2.7532121e-10 2.3093794e-11 9.9999976e-01]]
print(pred1.argmax())
print(x_test[0].shape)


