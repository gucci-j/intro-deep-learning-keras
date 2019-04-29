from keras import Model
from keras.layers import Input, Dense

import numpy as np

w1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
b1 = np.array([0.1, 0.2, 0.3])
w2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
b2 = np.array([0.1, 0.2])
w3 = np.array([[0.1, 0.3], [0.2, 0.4]])
b3 = np.array([0.1, 0.2])
weight_array = [w1, b1, w2, b2, w3, b3]

_input = Input(shape=(2, ))
_layer1 = Dense(units=3, activation='sigmoid')(_input)
_layer2 = Dense(units=2, activation='sigmoid')(_layer1)
_output = Dense(units=2)(_layer2)

model = Model(inputs=_input, outputs=_output)
model.summary() # モデルの状態をみる

model.set_weights(weight_array)
print(model.get_weights())

X = np.array([[1.0, 0.5]])
Y = model.predict(X)

print(Y)