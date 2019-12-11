import os
import cv2
import glob
import numpy as np
from keras.models import load_model

from functions import SAVED_MODEL_PATH, TRAINING_PATH, EVALUATE_PATH, get_latest_model, load_images

########################################################################################################################

model_name = get_latest_model()
model = load_model(model_name)
print('-' * 80)
print("Loaded model " + model_name)
# model.summary()

########################################################################################################################

print("Loading evaluation dataset ...")
X_test, y_test = load_images(EVALUATE_PATH, 68, 69)
print("... done!")
print(X_test.shape, y_test.shape)

y_predict = model.predict(X_test)[0]
y_predict = cv2.normalize(y_predict, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

mask = cv2.normalize(y_test[0], None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

cv2.imshow('original', X_test[0])
cv2.imshow('mask', mask)
cv2.imshow('predict', y_predict)
cv2.waitKey()

########################################################################################################################
