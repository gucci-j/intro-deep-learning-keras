from keras.datasets import cifar10
import matplotlib.pyplot as plt
from scipy.misc import toimage

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

n = 3
for i in range(n):
    image = toimage(x_train[i])
    plt.subplot(1, n, i + 1)
    plt.imshow(image)
    plt.axis('off')
plt.show()
