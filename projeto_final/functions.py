import os
import cv2
import glob
import numpy as np


########################################################################################################################
# Path
SAVED_MODEL_PATH = 'saved_models'
TRAINING_PATH = 'RHD_published_v2/training'
EVALUATE_PATH = 'RHD_published_v2/evaluation'


########################################################################################################################

def get_latest_model():
    return max(glob.glob(SAVED_MODEL_PATH + "/*"), key=os.path.getctime)


########################################################################################################################

def load_images(path: str, init: int = 0, size: int = None, shape=(256, 256)):
    X, y = [], []

    images_names = os.listdir(os.path.join(path, 'color'))
    if size is not None:
        images_names = images_names[init:size]
    else:
        images_names = images_names[init:]

    for idx, name in enumerate(images_names):

        image = cv2.imread(os.path.join(path, 'color', name))
        image = cv2.resize(image, shape)

        mask = cv2.imread(os.path.join(path, 'mask', name), cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, shape).reshape((shape[0], shape[1], 1))
        mask = cv2.normalize(mask, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        X.append(image)
        y.append(mask)

        if idx % 500 == 0:
            print("%d of %d" % (idx, len(images_names)))

    return np.array(X), np.array(y)


