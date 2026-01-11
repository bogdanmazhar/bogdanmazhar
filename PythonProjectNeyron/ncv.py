import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
x_train_values = tf.constant([20], dtype=tf.float32)
x_train_cost = tf.constant([10], dtype=tf.float32)
y_train_stations = tf.constant([400], dtype=tf.float32)
x_train = tf.stack([x_train_values, x_train_cost], axis=1)

model = keras.Sequential([
    layers.Dense(5, activation='relu', input_shape=(2,)),
    layers.Dense(2),
    layers.Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
model.fit(x_train, y_train_stations, epochs=500, verbose=0)
print('Обучение завершено!')

x_test_values = tf.constant([2], dtype=tf.float32)
x_test_cost = tf.constant([1], dtype=tf.float32)
x_test = tf.stack([x_test_values, x_test_cost], axis=1)
predictions = model.predict(x_test)

print('Спрос на електромобили:', predictions.flatten())