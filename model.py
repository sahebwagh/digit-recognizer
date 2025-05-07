import numpy as np
import tensorflow as tf

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# normalizing the data like  if we have 0-255 pixel img to like 0-1
x_train = x_train/255.0
x_test = x_test/255.0


tf.random.set_seed(1234)
model = tf.keras.Sequential(
    [
        tf.keras.layers.Flatten(input_shape=(28,28)),
        tf.keras.Dense(units=256, activation='relu'),
        tf.keras.Dense(units=128, activation='relu'),
        tf.keras.Dense(units=64, activation='relu'),
        tf.keras.Dense(units=10, activation='linear')
    ]
)


model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001),
              loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
              metrics=['accuracy']
              )

model.fit(x_train, y_train, epochs=40)



test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"\nTest accuracy: {test_acc:.4f}")

pred = model.predict(x_test[6].reshape(1,28,28))
print(pred)
print(np.argmax(pred))