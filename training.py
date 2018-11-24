from keras.layers import Input, Dense
from keras.models import Model
from keras import regularizers

encoding_size = 32 # size of encoding

img_input = Input(shape=(784,))
encoded_img = Dense(128, activation='relu')(img_input)
encoded_img = Dense(64, activation='relu')(encoded_img)
encoded_img = Dense(32, activation='relu')(encoded_img)

decoded_img = Dense(64, activation='relu')(encoded_img)
decoded_img = Dense(128, activation='relu')(decoded_img)
decoded_img = Dense(784, activation='sigmoid')(decoded_img)

autoencoder = Model(img_input, decoded_img)
# encoder = Model(img_input, encoded_img)
#
# encoded_input = Input(shape=(encoding_size,))
# output_layer = autoencoder.layers[-1]
# decoder = Model(encoded_input, output_layer(encoded_input))

autoencoder.compile(optimizer='adadelta',
                    loss='binary_crossentropy')

# prepare dataset
from keras.datasets import mnist
import numpy as np


(x_train, y_train), (x_test, y_test) = mnist.load_data()

train_filter = np.where((y_train == 2))
test_filter = np.where((y_test == 2))

x_train, y_train = x_train[train_filter], y_train[train_filter]
x_test, y_test = x_test[test_filter], y_test[test_filter]

x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255
x_train = x_train.reshape((len(x_train),
                           np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test),
                           np.prod(x_test.shape[1:])))

# train
autoencoder.fit(x_train, x_train,
                epochs=1023,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))

# encoder.save('encoder_weights.h5')
# decoder.save('decoder_weights.h5')
