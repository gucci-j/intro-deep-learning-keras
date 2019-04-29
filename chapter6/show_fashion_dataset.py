from keras.datasets import fashion_mnist
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

plt.figure()
plt.imshow(x_train[0], cmap='gray')
plt.show()