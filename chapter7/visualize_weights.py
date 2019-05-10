from keras import Model
from keras.layers import Input, Dense, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import array_to_img
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import math

def build_model():
    # モデル定義
    _input = Input(shape=(32, 32, 3))
    _hidden = Conv2D(filters=30, kernel_size=5, strides=(1, 1), padding='valid', activation='relu')(_input)
    _hidden = MaxPooling2D(pool_size=(2, 2))(_hidden)
    _hidden = Flatten()(_hidden)
    _hidden = Dense(100, activation='relu')(_hidden)
    _output = Dense(10, activation='softmax')(_hidden)
    model = Model(_input, _output)

    return model

def set_flag():
    # フラグの定義
    flag = {}
    flag['before_param'] = False
    flag['after_param'] = True

    return flag

def visualizer(weights, key):
    # 重みの可視化
    weights = weights[0] # 畳み込み層の重みを取ってくる
    weights = np.split(weights, 30, axis=3) # 各フィルターに分割する
    weight_image = []

    for weight in weights:
        weight = np.squeeze(weight, axis=3)
        weight = array_to_img(weight)
        weight = np.array(weight)
        weight_image.append(weight)
    
    img = combine_images(np.array(weight_image)) # 重み画像の合体
    Image.fromarray(img.astype(np.uint8)).save(key + '.png')

def combine_images(images):
    num = images.shape[0]
    width = int(math.sqrt(num))
    height = int(math.ceil(float(num) / width))
    shape = images.shape[1:]
    image = np.zeros((height * shape[0], width * shape[1], shape[2]),
                     dtype=images.dtype)
    for index, img in enumerate(images):
        i = int(index / width)
        j = index % width
        image[i * shape[0]:(i + 1) * shape[0], j * shape[1]:(j + 1) * shape[1], :] = img
    return image

def main():
    flag = set_flag()

    for key in flag.keys():
        model = build_model()
        model.load_weights(key + '.h5')
        weights = model.get_weights()
        visualizer(weights, key)

if __name__ == '__main__':
    main()