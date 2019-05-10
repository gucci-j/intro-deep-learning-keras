import matplotlib as mpl
mpl.use('Agg')
from keras import Model
from keras.layers import Input, Dense
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.datasets import cifar10
from keras.utils import to_categorical
import matplotlib.pyplot as plt

# モデル定義
_input = Input(shape=(32, 32, 3))
_hidden = Conv2D(filters=30, kernel_size=5, strides=(1, 1), padding='valid', activation='relu')(_input)
_hidden = MaxPooling2D(pool_size=(2, 2))(_hidden)
_hidden = GlobalAveragePooling2D()(_hidden)
_hidden = Dense(100, activation='relu')(_hidden)
_output = Dense(10, activation='softmax')(_hidden)

model = Model(_input, _output)
model.summary()

model.save_weights('before_gap_param.h5')

# データセットの読み込み
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float') / 255.
x_test = x_test.astype('float') / 255.
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# 学習設定
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
_results = model.fit(x=x_train, y=y_train, batch_size=100, epochs=150, verbose=1, validation_data=(x_test, y_test))

# 結果のプロット
loss = _results.history['loss']
val_loss = _results.history['val_loss']
acc = _results.history['acc']
val_acc = _results.history['val_acc']

model.save_weights('after_gap_param.h5')

plt.figure()
plt.plot(range(1, 150+1), loss, marker='.', label='train')
plt.plot(range(1, 150+1), val_loss, marker='.', label='test')
plt.legend(loc='best', fontsize=10)
plt.grid()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.savefig('loss_gap.png')

plt.clf()
plt.plot(range(1, 150+1), acc, marker='.', label='train')
plt.plot(range(1, 150+1), val_acc, marker='.', label='test')
plt.legend(loc='best', fontsize=10)
plt.grid()
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.savefig('acc_gap.png')
plt.close()