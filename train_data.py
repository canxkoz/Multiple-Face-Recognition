import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
from keras import callbacks
from keras.models import Sequential, model_from_yaml, load_model
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPool2D
from keras.optimizers import Adam, SGD
from keras.utils import np_utils, plot_model
from sklearn.model_selection import train_test_split
from keras.applications.resnet50 import preprocess_input, decode_predictions
import cv2


np.random.seed(7)
img_h, img_w = 128, 128
image_size = (128, 128)
nbatch_size = 2
nepochs = 30
nb_classes = 2
def load_data():
    path = 'dataset/train'
    files = os.listdir(path)
    images = []
    labels = []
    for f in files:
        img_array = cv2.imread(os.path.join(path,f))
        new_array = cv2.resize(img_array, (img_h, img_w))
        images.append(new_array)

        if 'Can' in f:
            labels.append(0)
        else:
            labels.append(1)

    data = np.array(images)
    data = np.array(data).reshape(-1, img_h, img_h, 3)
    labels = np.array(labels)

    labels = np_utils.to_categorical(labels, 2)
    return data, labels


model = Sequential()
# 32:output dimension of the convolutional layer
model.add(Conv2D(32, kernel_size=(3, 3), input_shape=(img_h, img_h, 3), activation='relu', padding='same'))
model.add(MaxPool2D(2,2))
model.add(Dropout(0.3))

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(MaxPool2D(2,2))
model.add(Dropout(0.3))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(MaxPool2D(2,2))
model.add(Dropout(0.5))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(MaxPool2D(2,2))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(2, activation='softmax'))

model.summary()

print("compile.......")
sgd = Adam(lr=0.0003)
model.compile(loss='binary_crossentropy',optimizer=sgd, metrics=['accuracy'])
#The Adam optimizer is used, the learning rate is 0.0003, the default is 0.0001

# Loading the dataset
print("load_data......")
images, lables = load_data()
images = images/255
x_train, x_test, y_train, y_test = train_test_split(images, lables, test_size=0.1)
print(x_train.shape,y_train.shape)

# Viewing training on TensorBoard
# print("train.......")
# tbCallbacks = callbacks.TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True, write_images=True)

model.fit(x_train, y_train, batch_size=nbatch_size, epochs=nepochs, verbose=1, validation_data=(x_test, y_test))

# Evaluating the model
print("evaluate......")
score, accuracy = model.evaluate(x_test, y_test, batch_size=nbatch_size)
print('score:', score, 'accuracy:', accuracy)

# Saving the mofel and making the reaining process much easier and mre practial. Saving the weights separately： from
# keras.models import model_from_yaml, load_model
yaml_string = model.to_yaml()
with open('keras_model/S_H.yaml', 'w') as outfile:
    outfile.write(yaml_string)
model.save_weights('keras_model/S_H.h5')



