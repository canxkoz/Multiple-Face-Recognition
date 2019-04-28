import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
from keras.models import Sequential, model_from_yaml, load_model
from keras.optimizers import Adam, SGD
import cv2

img_h, img_w = 128,128
with open('keras_model/S_H.yaml') as yamlfile:
    loaded_model_yaml = yamlfile.read()
model = model_from_yaml(loaded_model_yaml)
model.load_weights('keras_model/S_H.h5')

sgd = Adam(lr=0.0003)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

images = []
path = 'dataset/test'
for f in os.listdir(path):
    img_array = cv2.imread(os.path.join(path, f))
    new_array = cv2.resize(img_array, (img_h, img_w))

    data = np.array(new_array)
    data = np.array(data).reshape(-1, img_h, img_h, 3)

    result = model.predict_classes(data, verbose=0)

    print(f, result[0])