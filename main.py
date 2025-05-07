# import os
# import numpy as np
# import matplotlib.pyplot as plt
# import tensorflow as tf
# from PIL.ImageOps import grayscale
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.activations import linear, relu, sigmoid
# from tensorflow.python.ops.numpy_ops.np_array_ops import reshape

# mnist = tf.keras.datasets.mnist
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# plt.imshow(x_test[6], cmap='gray')
# plt.show()
# normalizing the data like  if we have 0-255 pixel img to like 0-1
# x_train = tf.keras.utils.normalize(x_train, axis=1)
# x_test = tf.keras.utils.normalize(x_test, axis=1)
#
#
#
# model = Sequential()
# model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
# model.add(Dense(units=128, activation='relu', name='layer_1'))
# model.add(Dense(units=128, activation='relu', name='layer_2'))
# model.add(Dense(units=10, activation='softmax', name='layer_3'))
#
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#
# model.fit(x_train, y_train, epochs=3)
#
# prediction = model.predict(x_test[0].reshape(1,28,28))
# print(f"{prediction}")
# print(np.argmax(prediction))
#
#












#
# """used to save a trained model for later use
# It works for saving the entire model, including the architecture, weights, and training configuration
#  (like optimizer and loss"""
#
# # model.save('handwritten_model.keras')
# # model = tf.keras.models.load_model('handwritten_model.keras')
#
# # Evaluate the model
# loss, accuracy = model.evaluate(x_test, y_test)
#
# print(f"Test Loss: {loss}")
# print(f"Test Accuracy: {accuracy * 100:.2f}%")




