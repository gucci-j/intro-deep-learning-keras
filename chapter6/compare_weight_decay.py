from keras import Model, optimizers, initializers, regularizers
from keras.layers import Input, Dense, Activation
from keras.layers.normalization import BatchNormalization
from keras.utils import to_categorical
from keras.datasets import fashion_mnist
import matplotlib.pyplot as plt

# パラメータ + ハイパーパラメータ
img_shape = (28 * 28, )
hidden_dim = 100
output_dim = 10
batch_size = 128
learning_rate = 0.01
epochs = 15
_init = initializers.RandomNormal(mean=0.0, stddev=0.01, seed=None)
_wd = regularizers.l2(0.1)

def build_model(wd):
    # モデルを定義する
    if wd is True:
        _input = Input(shape=img_shape)
        _hidden = Dense(hidden_dim, kernel_initializer=_init, kernel_regularizer=_wd)(_input)
        _hidden = BatchNormalization()(_hidden)
        _hidden = Activation('relu')(_hidden)
        _hidden = Dense(hidden_dim, kernel_initializer=_init, kernel_regularizer=_wd)(_hidden)
        _hidden = BatchNormalization()(_hidden)
        _hidden = Activation('relu')(_hidden)
        _hidden = Dense(hidden_dim, kernel_initializer=_init, kernel_regularizer=_wd)(_hidden)
        _hidden = BatchNormalization()(_hidden)
        _hidden = Activation('relu')(_hidden)
        _hidden = Dense(hidden_dim, kernel_initializer=_init, kernel_regularizer=_wd)(_hidden)
        _hidden = BatchNormalization()(_hidden)
        _hidden = Activation('relu')(_hidden)
        _output = Dense(output_dim, activation='softmax')(_hidden)

        model = Model(inputs=_input, outputs=_output)
        return model

    else:
        _input = Input(shape=img_shape)
        _hidden = Dense(hidden_dim, kernel_initializer=_init)(_input)
        _hidden = BatchNormalization()(_hidden)
        _hidden = Activation('relu')(_hidden)
        _hidden = Dense(hidden_dim, kernel_initializer=_init)(_hidden)
        _hidden = BatchNormalization()(_hidden)
        _hidden = Activation('relu')(_hidden)
        _hidden = Dense(hidden_dim, kernel_initializer=_init)(_hidden)
        _hidden = BatchNormalization()(_hidden)
        _hidden = Activation('relu')(_hidden)
        _hidden = Dense(hidden_dim, kernel_initializer=_init)(_hidden)
        _hidden = BatchNormalization()(_hidden)
        _hidden = Activation('relu')(_hidden)
        _output = Dense(output_dim, activation='softmax')(_hidden)

        model = Model(inputs=_input, outputs=_output)
        return model



def load_data():
    # データを読み込む
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x_train = x_train[:30000]
    y_train = y_train[:30000]
    x_train = x_train.reshape(30000, 784)
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype('float') / 255.
    x_test = x_test.astype('float') / 255.
    print(f'Before: {y_train.shape}')
    print(f'y_train[0]: {y_train[0]}')
    y_train = to_categorical(y_train, num_classes=output_dim)
    print(f'After: {y_train.shape}')
    print(f'y_train[0]: {y_train[0]}')
    y_test = to_categorical(y_test, num_classes=output_dim)

    return x_train, y_train, x_test, y_test


def set_flag():
    # バッチ正規化フラグの定義
    flag = {}
    flag['With Weight decay'] = True
    flag['Without Weight decay'] = False

    return flag


def main():
    x_train, y_train, x_test, y_test = load_data()
    flag = set_flag()

    results = {}
    for key in flag.keys():
        print(f'---Now running: {key} model---')
        model = build_model(flag[key])
        model.compile(optimizer=optimizers.SGD(lr=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
        results[key] = model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))

    plt.figure()
    for key in flag.keys():
        acc = results[key].history['acc']
        val_acc = results[key].history['val_acc']
        plt.plot(range(1, epochs+1), acc, marker='.', label='train')
        plt.plot(range(1, epochs+1), val_acc, marker='.', label='test')
        plt.legend(loc='best', fontsize=10)
        plt.grid()
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.savefig('acc_' + key + '.png')
        plt.clf()


if __name__ == '__main__':
    main()
