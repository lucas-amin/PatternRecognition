import os
import cv2
import glob
import time
import keras
import datetime
import numpy as np
from keras.models import load_model

from functions import SAVED_MODEL_PATH, TRAINING_PATH, EVALUATE_PATH, get_latest_model, load_images

########################################################################################################################

# Training parameters
batch_size = 16
epochs = 6

TIME = 3

########################################################################################################################

print("Loading training dataset ...")
X_train, y_train = load_images(TRAINING_PATH, 0, 10000)
# X_train, y_train = load_images(TRAINING_PATH, 5000 * TIME, 5000 * (TIME + 1))
print("... done!")
print(X_train.shape, y_train.shape)

print("Loading evaluation dataset ...")
X_test, y_test = load_images(EVALUATE_PATH, 0, 500)
print("... done!")
print(X_test.shape, y_test.shape)


########################################################################################################################

model_name = get_latest_model()
model = load_model(model_name)
print('-' * 80)
print("Loaded model " + model_name)
# model.summary()


########################################################################################################################

print('-' * 80)
print("Starting to train..")
start_time = time.time()
model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(X_test, y_test),
          shuffle=True)
print("Fitting duration: " + str(time.time() - start_time))

new_model_name = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S.h5")
model.save(os.path.join(SAVED_MODEL_PATH, new_model_name))
print("Saved model " + new_model_name)

# Score trained model.
scores = model.evaluate(X_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
