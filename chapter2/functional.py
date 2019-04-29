from keras import Model
from keras.layers import Input, Dense

import numpy as np

_input = Input(shape=(2, ))
_output = Dense(units=1)(_input)

model = Model(inputs=_input, outputs=_output)
model.summary() # モデルの状態をみる

model.set_weights([np.array([[0.5], [0.5]]), np.array([-0.7])])

X = np.array([[0,0], [0,1], [1,0], [1,1]])
Y = np.array([[0], [0], [0], [1]])
print(model.get_weights())
Y_ = model.predict(X)

print(Y_)
Y_[Y_ <= 0] = False
Y_[Y_ > 0] = True
print(Y_)
print(f'Results: {Y == Y_}')