import matplotlib as mpl
mpl.use('Agg')
from keras.applications.mobilenet import MobileNet
from keras import Model
from keras.layers import Input, Dense, Flatten
from keras.datasets import cifar10
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from PIL import Image
from keras.preprocessing import image
import numpy as np

_input = Input(shape=(32, 32, 3))
_hidden = MobileNet(include_top=False, pooling='avg')(_input)
_output = Dense(10, activation='softmax')(_hidden)
model = Model(_input, _output)
model.summary()

"""
def resize(x):
    x_list = []
    for i in range(x.shape[0]):
        img = image.array_to_img(x[i, :, :, :].reshape(32, 32, -1))
        img = img.resize(size=(224, 224), resample=Image.LANCZOS)
        x_list.append(image.img_to_array(img))
    return np.array(x_list).astype('float') / 255.
"""

# データセットの読み込み
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# x_train = resize(x_train)
# x_test = resize(x_test)
x_train = x_train.astype('float') / 255.
x_test = x_test.astype('float') / 255.
print(f'x_train shape: {x_train.shape}')
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# 学習設定
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 事前学習済みモデルのテスト
print(model.evaluate(x=x_test, y=y_test, batch_size=128))

# 転移学習してみる
_results = model.fit(x=x_train, y=y_train, batch_size=128, epochs=3, verbose=1, validation_data=(x_test, y_test))

model.save_weights('pretrained_after_param.h5')