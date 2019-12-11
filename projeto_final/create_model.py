import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, UpSampling2D, Lambda
import os
import datetime

# Reference for parameters:
# https://github.com/tranquanghuy0801/HandSegNet/blob/master/main.py

model = Sequential()
model.add(Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(256, 256, 3)))
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(4, 4), strides=2, padding='same'))

model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(4, 4), strides=2, padding='same'))

model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))

model.add(Conv2D(1, (1, 1), padding='same'))
model.add(UpSampling2D((4, 4)))


# Importing Keras in lambda function to prevent a bug
def lambda_argmax(x):
    from keras import backend
    return backend.softmax(x)


def lambda_cast(x):
    from keras import backend
    return backend.cast(x, "float")


# Argmax layer
# model.add(Lambda(lambda_argmax))
# model.add(Lambda(lambda_cast))

opt = keras.optimizers.Adam(lr=0.0001, decay=1e-6)

# Let's train the model using RMSprop
model.compile(loss='mean_squared_error',
              optimizer=opt,
              metrics=['accuracy'])

model.summary()

model_name = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S.h5")
save_dir = os.path.join(os.getcwd(), 'saved_models')
model.save(os.path.join(save_dir, model_name))
