from keras.datasets import cifar10
from keras.utils import to_categorical
from src.cnn import build_model

from sklearn.model_selection import KFold, StratifiedKFold
import numpy as np

def main(stratified=False):
    # データセットの読み込み
    (x_train, y_train), (_, _) = cifar10.load_data()
    x_train = x_train.astype('float') / 255.
    _y_train = to_categorical(y_train, num_classes=10)

    # k-fold CV
    _history = []
    kf = None
    if stratified is False:
        kf = KFold(n_splits=5, random_state=1234)
    else:
        kf = StratifiedKFold(n_splits=5, random_state=1234)
    for train_index, val_index in kf.split(x_train, y_train):
        model = build_model()
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(x=x_train[train_index], y=_y_train[train_index], batch_size=100, epochs=10, verbose=1)
        _history.append(model.evaluate(x=x_train[val_index], y=_y_train[val_index], batch_size=100))

    _history = np.asarray(_history)
    loss = np.mean(_history[:, 0])
    acc = np.mean(_history[:, 1])
    print(f'loss: {loss} ± {np.std(_history[:, 0])} | acc: {acc} ± {np.std(_history[:, 1])}')

if __name__ == '__main__':
    main()