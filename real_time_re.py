import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
from keras.models import Sequential, model_from_yaml, load_model
from keras.optimizers import Adam, SGD
import cv2

detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
img_h, img_w = 128,128
with open('keras_model/S_H.yaml') as yamlfile:
    loaded_model_yaml = yamlfile.read()
model = model_from_yaml(loaded_model_yaml)
model.load_weights('keras_model/S_H.h5')

sgd = Adam(lr=0.0005)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

capture = cv2.VideoCapture(0)

while True:
    ret, frame = capture.read()
    faces = detector.detectMultiScale(frame, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        ROI = frame[y:y + h, x:x + w]
        for f in faces:
            new_array = cv2.resize(ROI, (img_h, img_w))
            data = np.array(new_array)
            data = np.array(data).reshape(-1, img_h, img_h, 3)
            result = model.predict_classes(data, verbose=0)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            if result[0] == 0:
                cv2.putText(frame, 'Can', (x, y), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 2)
            elif result[0] == 1:
                cv2.putText(frame, 'Andrew', (x, y), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 2)
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

capture.release()
cv2.destroyAllWindows()
