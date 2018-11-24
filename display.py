from keras.models import load_model
from PIL import Image
from random import random
import cv2 as cv

encoder = load_model('encoder_weights.h5')
decoder = load_model('decoder_weights.h5')

# get mnist

from keras.datasets import mnist
import numpy as np

# we don't care about labels
(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255
x_train = x_train.reshape((len(x_train),
                           np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test),
                           np.prod(x_test.shape[1:])))

encoded = encoder.predict(x_test)
decoded = decoder.predict(encoded)

input = np.zeros((1, 16))
#out = decoder.predict(test).reshape(28, 28) * 255

for i in range(1000):
    for j in range(16):
        input[0][j] = (input[0][j] + random()) % 50
    out = decoder.predict(input).reshape(28, 28) * 255
    cv.imwrite("generated_img/%d.png" % (i), cv.resize(out, dsize=(280, 280), interpolation=cv.INTER_CUBIC))

import os

os.chdir("generated_img")
os.system("del video.avi")
os.system("del out.gif")
os.system("ffmpeg -f image2 -i %d.png video.avi")
os.system("ffmpeg -i video.avi -pix_fmt rgb24 out.gif")
os.system("del *.png")
os.system("del video.avi")

# # display
# import matplotlib.pyplot as plt
#
# n = 100  # how many digits we will display
# plt.figure(figsize=(20, 4))
# for i in range(n):
#     # display original
#     ax = plt.subplot(2, n, i + 1)
#     plt.imshow(x_test[i].reshape(28, 28))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
#
#     # display reconstruction
#     ax = plt.subplot(2, n, i + 1 + n)
#     plt.imshow(decoded[i].reshape(28, 28))
#     plt.gray()
#     ax.get_xaxis().set_visible(False)
#     ax.get_yaxis().set_visible(False)
# plt.show()
