from keras import Model
from keras.layers import Input, Dense
from keras.optimizers import SGD
from keras.datasets import mnist
from keras.utils import to_categorical

import matplotlib.pyplot as plt

# パラメータ + ハイパーパラメータ
img_shape = (28 * 28, )
hidden_dim = 100
output_dim = 10
batch_size = 100
learning_rate = 0.1
epochs = 17

# モデルを定義する
_input = Input(shape=img_shape)
_hidden = Dense(hidden_dim, activation='sigmoid')(_input)
_output = Dense(output_dim, activation='softmax')(_hidden)

model = Model(inputs=_input, outputs=_output)
model.summary()

# データを読み込む
(x_train, y_train), (x_test, y_test) = mnist.load_data()
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

# 学習の実行
sgd = SGD(lr=learning_rate)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

_results = model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))

# 結果の確認
print(f'Keys: {_results.history.keys()}')

loss = _results.history['loss']
val_loss = _results.history['val_loss']
acc = _results.history['acc']
val_acc = _results.history['val_acc']

plt.figure()
plt.plot(range(1, epochs+1), loss, marker='.', label='train')
plt.plot(range(1, epochs+1), val_loss, marker='.', label='test')
plt.legend(loc='best', fontsize=10)
plt.grid()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.savefig('loss.png')

plt.clf()
plt.plot(range(1, epochs+1), acc, marker='.', label='train')
plt.plot(range(1, epochs+1), val_acc, marker='.', label='test')
plt.legend(loc='best', fontsize=10)
plt.grid()
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.savefig('acc.png')
plt.close()