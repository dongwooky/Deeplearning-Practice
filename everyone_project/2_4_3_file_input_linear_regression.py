import tensorflow as tf
import numpy as np

xy = np.loadtxt('/Users/82106/Documents/Git_project/deeplearning_practice/everyone_project/data/data-01-test-score.csv', delimiter=',', dtype=np.float32)
x_data=xy[:, 0:-1]
y_data=xy[:, [-1]]

print(x_data, "\nx_data shape:", x_data.shape)
print(y_data, "\ny_data shape:", y_data.shape)

tf.model = tf.keras.Sequential()
tf.model.add(tf.keras.layers.Dense(units=1 ,input_dim=3, activation='linear'))
tf.model.summary()

tf.model.compile(loss='mse', optimizer=tf.keras.optimizers.SGD(lr=1e-5))
history=tf.model.fit(x_data, y_data, epochs=2000)

print("Your score will be ", tf.model.predict([[100, 70, 101]]))
print("Other scores will be ", tf.model.predict([[60, 70, 110], [90, 100, 80]]))