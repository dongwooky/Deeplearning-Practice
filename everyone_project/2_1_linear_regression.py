import numpy as np
import tensorflow as tf

x_train=[1,2,3,4]
y_train=[0,-1,-2,-3]

tf.model=tf.keras.Sequential()
tf.model.add(tf.keras.layers.Dense(units=1, input_dim=1))

sgd=tf.keras.optimizers.SGD(lr=0.1)
tf.model.compile(loss='mse', optimizer=sgd)

tf.model.summary()

tf.model.fit(x_train, y_train, epochs=200)

y_predict=tf.model.predict(np.array([5,4]))
print(y_predict)