from keras import Model
from keras.layers import Input, Dense, Activation
import numpy as np
from keras.datasets import mnist
import pickle

def load_weight():
    with open('sample_weight.pkl', 'rb') as f:
        weights = pickle.load(f)
        weight_array = [weights['W1'], weights['b1'], 
                        weights['W2'], weights['b2'],
                        weights['W3'], weights['b3']]
        
        return weight_array

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape)
print(y_train.shape)

# 一次元配列にする
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

# 正規化処理
x_train = x_train.astype('float')
x_test = x_test.astype('float')
x_train /= 255.
x_test /= 255.

print(x_train.shape)

_input = Input(shape=(784, ))
_hidden = Dense(units=50, activation='sigmoid')(_input)
_hidden = Dense(units=100, activation='sigmoid')(_hidden)
_hidden = Dense(units=10)(_hidden)
_output = Activation('softmax')(_hidden)

model = Model(inputs=_input, outputs=_output)
model.summary() # モデルの状態をみる

model.set_weights(load_weight())

_y_test = model.predict(x_test)
_y_test = np.argmax(_y_test, axis=1)

print(f'Accuracy: {np.sum(y_test == _y_test) / len(y_test)}')