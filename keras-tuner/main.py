from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from kerastuner.tuners import RandomSearch

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
import numpy as np

def build_model(hp) -> Model:
    # モデル定義
    _input = Input(shape=(32, 32, 3))
    _hidden = Conv2D(filters=hp.Range('filters', min_value=10, 
                    max_value=40, step=10), 
                    kernel_size=hp.Range('kernel_size', min_value=2,
                    max_value=5, step=1), 
                    strides=(1, 1), padding='valid', activation='relu')(_input)
    _hidden = MaxPooling2D(pool_size=(2, 2))(_hidden)
    _hidden = Flatten()(_hidden)
    _hidden = Dense(units=hp.Range('units', min_value=50,
                    max_value=200, step=50), 
                    activation='relu')(_hidden)
    _output = Dense(10, activation='softmax')(_hidden)
    model = Model(_input, _output)
    model.compile(optimizer=Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
        loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def load_data():
    # データセットの読み込み
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.astype('float') / 255.
    x_test = x_test.astype('float') / 255.
    y_train = to_categorical(y_train, num_classes=10)
    y_test = to_categorical(y_test, num_classes=10)

    return x_train, y_train, x_test, y_test


def main():
    tuner = RandomSearch(build_model, objective='val_accuracy',
        max_trials=5, executions_per_trial=1, directory='tuning', project_name='log')
    
    x_train, y_train, x_test, y_test = load_data()

    tuner.search_space_summary()
    tuner.search(x_train, y_train, epochs=10, validation_data=(x_test, y_test))
    tuner.results_summary()

if __name__ == '__main__':
    main()