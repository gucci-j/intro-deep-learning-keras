from keras import Model, optimizers
from keras.layers import Input, Dense
from keras.utils import to_categorical
from keras.datasets import fashion_mnist
import matplotlib.pyplot as plt

# パラメータ + ハイパーパラメータ
img_shape = (28 * 28, )
hidden_dim = 100
output_dim = 10
batch_size = 128
learning_rate = 0.1
epochs = 15

def build_model():
    # モデルを定義する
    _input = Input(shape=img_shape)
    _hidden = Dense(hidden_dim, activation='sigmoid')(_input)
    _hidden = Dense(hidden_dim, activation='sigmoid')(_hidden)
    _hidden = Dense(hidden_dim, activation='sigmoid')(_hidden)
    _hidden = Dense(hidden_dim, activation='sigmoid')(_hidden)
    _output = Dense(output_dim, activation='softmax')(_hidden)

    model = Model(inputs=_input, outputs=_output)
    return model


def load_data():
    # データを読み込む
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    x_train = x_train.reshape(60000, 784)
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


def set_optimizers():
    # 最適化アルゴリズムの定義
    optim = {}
    optim['SGD'] = optimizers.SGD(lr=learning_rate)
    optim['Adagrad'] = optimizers.Adagrad(lr=learning_rate)
    optim['Adadelta'] = optimizers.Adadelta(lr=learning_rate)
    optim['Adam'] = optimizers.Adam()
    optim['Nadam'] = optimizers.Nadam()

    return optim


def main():
    x_train, y_train, x_test, y_test = load_data()
    optim = set_optimizers()

    results = {}
    for key in optim.keys():
        print(f'---Now running: {key} model---')
        model = build_model()
        model.compile(optimizer=optim[key], loss='categorical_crossentropy', metrics=['accuracy'])
        results[key] = model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))

    plt.figure()
    for key in optim.keys():
        loss = results[key].history['loss']
        plt.plot(range(1, epochs+1), loss, marker='.', label=key)
    plt.legend(loc='best', fontsize=10)
    plt.grid()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig('loss.png')


if __name__ == '__main__':
    main()
