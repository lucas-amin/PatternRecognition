""" Basic example showing samples from the dataset.

    Chose the set you want to see and run the script:
    > python view_samples.py

    Close the figure to see the next sample.
"""
from __future__ import print_function, unicode_literals

import pickle
import os

import cv2
import numpy as np
import scipy.misc
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import imageio

# chose between training and evaluation set
# set = 'training'
set = 'evaluation'

# auxiliary function
def depth_two_uint8_to_float(top_bits, bottom_bits):
    """ Converts a RGB-coded depth into float valued depth. """
    depth_map = (top_bits * 2**8 + bottom_bits).astype('float32')
    depth_map /= float(2**16 - 1)
    depth_map *= 5.0
    return depth_map

# load annotations of this set
with open(os.path.join(set, 'anno_%s.pickle' % set), 'rb') as fi:
    anno_all = pickle.load(fi)

# iterate samples of the set
for sample_id, anno in anno_all.items():
    # load data
    image = cv2.imread(os.path.join(set, 'color', '%.5d.png' % sample_id))
    mask = cv2.imread(os.path.join(set, 'mask', '%.5d.png' % sample_id))
    mask = cv2.normalize(mask, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    X.append(image)
    y.append(mask)

    cv2.imshow("image", image)
    cv2.imshow("mask", mask)
    cv2.waitKey(0)


# ----------------------------------

img_height = img_width = 32

# Instanciando o modelo Resnet50V2
base_model = applications.resnet_v2.ResNet50V2(weights=None, include_top=False, input_shape=(img_height, img_width, 3))

# Aplicando pooling e dropout no output
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.7)(x)

# Gerando camada de output e instanciando o modelo
predictions = Dense(num_classes, activation='softmax')(x)
model2 = Model(inputs=base_model.input, outputs=predictions)

from keras.optimizers import SGD, Adam

model2.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

start_time = time.time()

model2.fit(x_train, y_train,
      batch_size=batch_size,
      epochs=epochs,
      validation_data=(x_test, y_test),
      shuffle=True)

print("Fitting duration: " + str(time.time() - start_time))

acc = model2.evaluate(x_test, y_test)
print("Test accuracy:" + str(acc[1]))
print("Loss:" + str(acc[0]))