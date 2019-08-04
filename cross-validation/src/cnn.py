from keras import Model
from keras.layers import Input, Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D

def build_model() -> Model:
    # モデル定義
    _input = Input(shape=(32, 32, 3))
    _hidden = Conv2D(filters=30, kernel_size=5, strides=(1, 1), padding='valid', activation='relu')(_input)
    _hidden = MaxPooling2D(pool_size=(2, 2))(_hidden)
    _hidden = Flatten()(_hidden)
    _hidden = Dense(100, activation='relu')(_hidden)
    _output = Dense(10, activation='softmax')(_hidden)

    model = Model(_input, _output)
    model.summary()

    return model