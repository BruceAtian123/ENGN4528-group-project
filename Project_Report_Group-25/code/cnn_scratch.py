import os
from sklearn.datasets import load_files
import matplotlib.pyplot as plt
import csv
import random
import tensorflow as tf
import pandas as pd
import cv2
import numpy as np
from glob import glob
import keras
from PIL import Image
from keras.preprocessing.image import ImageDataGenerator
from keras_tqdm import TQDMNotebookCallback
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.callbacks import Callback
from keras.models import load_model
import h5py
from keras.preprocessing.image import *
from keras.models import model_from_json
from keras.optimizers import *
from keras import applications
from tqdm import tqdm

'''
=====================================================================================================================
This part is using for divide cats and dogs image into train, validation and test data set 
=====================================================================================================================
'''


# with open('annotations/list.txt') as file:
#     cats = []
#     dogs = []
#     all_cats = {}
#     all_dogs = {}
#     cat_name = dict()
#     dog_name = dict()
#     for lines in file.readlines():
#         lines = lines.split(' ')
#         name = lines[0].split('_')
#         if lines[-2] == '1':
#             cats.append(lines[0])
#             if len(name) == 2:
#                 if name[0] in cat_name:
#                     cat_name[name[0]] = cat_name[name[0]] + 1
#                     all_cats[name[0]] = all_cats[name[0]] + ', ' + lines[0]
#                 else:
#                     cat_name[name[0]] = 1
#                     all_cats[name[0]] = lines[0]
#             elif len(name) > 2:
#                 if (str(name[0]) + '_' + str(name[1])) in cat_name:
#                     cat_name[str(name[0]) + '_' + str(name[1])] = cat_name[str(name[0]) + '_' + str(name[1])] + 1
#                     all_cats[str(name[0]) + '_' + str(name[1])] = all_cats[str(name[0]) + '_' + str(name[1])] + ', ' + lines[0]
#                 else:
#                     cat_name[(str(name[0]) + '_' + str(name[1]))] = 1
#                     all_cats[str(name[0]) + '_' + str(name[1])] = lines[0]
#         elif lines[-2] == '2':
#             dogs.append(lines[0])
#             if name[0] in dog_name:
#                 dog_name[name[0]] = dog_name[name[0]] + 1
#                 all_dogs[name[0]] = all_dogs[name[0]] + ', ' + lines[0]
#             else:
#                 dog_name[name[0]] = 1
#                 all_dogs[name[0]] = lines[0]
#         elif len(name) > 2:
#             if (str(name[0]) + '_' + str(name[1])) in dog_name:
#                 dog_name[str(name[0]) + '_' + str(name[1])] = dog_name[str(name[0]) + '_' + str(name[1])] + 1
#                 all_dogs[str(name[0]) + '_' + str(name[1])] = all_dogs[str(name[0]) + '_' + str(name[1])] + ', ' + lines[0]
#             else:
#                 dog_name[str(name[0]) + '_' + str(name[1])] = 1
#                 all_dogs[str(name[0]) + '_' + str(name[1])] = lines[0]
#
# cats_image = []
# dogs_image = []
# cats_keys = all_cats.keys()
# dogs_keys = all_dogs.keys()
#
#
# for ele in cats_keys:
#     all_cats[ele] = all_cats[ele].split(', ')
#     for filename in all_cats[ele]:
#         img = plt.imread(os.getcwd() + '/' + '/images/' + filename + '.jpg')
#         # img = plt.imread(os.getcwd() + '/' + '/annotations/trimaps/' + filename + '.png')
#         a = random.random()
#         if a <= 0.5:
#             plt.imsave(os.getcwd() + '/venv/data/train/cats/' + filename + '.jpg', img)
#             # plt.imsave(os.getcwd() + '/venv/data/train/cats_shape/' + filename + '.png', img)
#         elif a <= 0.75:
#             plt.imsave(os.getcwd() + '/venv/data/validation/cats/' + filename + '.jpg', img)
#             # plt.imsave(os.getcwd() + '/venv/data/validation/cats_shape/' + filename + '.png', img)
#         else:
#             plt.imsave(os.getcwd() + '/venv/data/test/cats/' + filename + '.jpg', img)
#             # plt.imsave(os.getcwd() + '/venv/data/testing/cats_shape/' + filename + '.png', img)
#
# for ele in dogs_keys:
#     all_dogs[ele] = all_dogs[ele].split(', ')
#     for filename in all_dogs[ele]:
#         img = plt.imread(os.getcwd() + '/' + '/images/' + filename + '.jpg')
#         # img = plt.imread(os.getcwd() + '/' + '/annotations/trimaps/' + filename + '.png')
#         a = random.random()
#         if a <= 0.5:
#             plt.imsave(os.getcwd() + '/venv/data/train/dogs/' + filename + '.jpg', img)
#             # plt.imsave(os.getcwd() + '/venv/data/train/dogs_shape/' + filename + '.png', img)
#         elif a <= 0.75:
#             plt.imsave(os.getcwd() + '/venv/data/validation/dogs/' + filename + '.jpg', img)
#             # plt.imsave(os.getcwd() + '/venv/data/validation/dogs_shape/' + filename + '.png', img)
#         else:
#             plt.imsave(os.getcwd() + '/venv/data/test/dogs/' + filename + '.jpg', img)
#             # plt.imsave(os.getcwd() + '/venv/data/testing/dogs_shape/' + filename + '.png', img)


'''
==================================================== main body =====================================================
'''


class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))


# define function to load train, test, and validation datasets
def load_dataset(path):
    data = load_files(path)
    dog_files = np.array(data['filenames'])
    dog_targets = np_utils.to_categorical(np.array(data['target']), 12)
    return dog_files, dog_targets


def predict_img(model, img_path, target_size):
    img = Image.open(img_path)
    if img.size != target_size:
        img = img.resize(target_size)

    x = image.img_to_array(img)
    x = x * 1./255
    x = np.expand_dims(x, axis=0)
    preds = model.predict(x)
    return preds[0]


batch = 32
train_datagen = ImageDataGenerator(rescale=1/255.,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True)

val_datagen = ImageDataGenerator(rescale=1 / 255.)
train_generator = train_datagen.flow_from_directory('/Users/jaychan/PycharmProjects/4528project/venv/data/images/train',
                                                    target_size=(224, 224),
                                                    batch_size=batch,
                                                    class_mode='categorical')

valid_generator = val_datagen.flow_from_directory('/Users/jaychan/PycharmProjects/4528project/venv/data/images/validation/',
                                                  target_size=(224, 224),
                                                  batch_size=batch,
                                                  class_mode='categorical')

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(224, 224, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
# model.add(Dropout(0.5))
model.add(Dense(37, activation='softmax'))

epochs = 10
learning_rate = 0.01
decay = learning_rate / epochs
sgd = SGD(lr=learning_rate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

model.summary()

history = LossHistory()

early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss',
                                               min_delta=0,
                                               patience=2,
                                               verbose=0,
                                               mode='auto')

fitted_model = model.fit_generator(train_generator,
                                   steps_per_epoch=int(7394 * 1 - 0.5) // batch,
                                   epochs=epochs,
                                   validation_data=valid_generator,
                                   validation_steps=int(7394 * 0.25) // batch,
                                   callbacks=[TQDMNotebookCallback(leave_inner=True, leave_outer=True), early_stopping,
                                              history])


model.save('./models/new_model_cla_mode.h5')


# model_json = model.to_json()
# with open('./data_model2.json', 'w') as jf:
#     jf.write(model_json)
#
losses, val_losses = history.losses, history.val_losses
fig = plt.figure(figsize=(15, 5))
plt.plot(fitted_model.history['loss'], 'g', label="train losses")
plt.plot(fitted_model.history['val_loss'], 'r', label="valid losses")
plt.grid(True)
plt.title('Training loss vs. Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

losses, val_losses = history.losses, history.val_losses
fig = plt.figure(figsize=(15, 5))
plt.plot(fitted_model.history['acc'], 'g', label="train acc")
plt.plot(fitted_model.history['val_acc'], 'r', label="valid acc")
plt.grid(True)
plt.title('Training acc vs. Validation acc')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()


# test_files, test_targets = load_dataset(os.getcwd() + '/venv/data1/test')
# test_datagen = ImageDataGenerator(rescale=1 / 255.)
# test_generator = test_datagen.flow_from_directory(
#                                                 './venv/data1/test/',
#                                                 target_size=(224, 224),
#                                                 batch_size=20,
#                                                 class_mode='categorical')
# js = open('data_model3.json', 'r')
# load_js = js.read()
# js.close()
# load_m = model_from_json(load_js)
# load_m.load_weights(os.getcwd() + '/models/model3_b.h5')
# load_m.compile(optimizer='sgd',
#                loss='binary_crossentropy',
#                metrics=['accuracy'])
# test_loss, test_acc = load_m.evaluate_generator(test_generator, steps=50)
# predIdxs = load_m.predict_generator(test_generator, steps=50)
# predIdxs = np.argmax(predIdxs, axis=1)
# print('test acc: ', test_acc)
# print('test loss:', test_loss)
# print(predict_img(load_m, './venv/data1/test/001.Abyssinian/Abyssinian_18.jpg', (150, 150)))
# print(predict_img(load_m, './venv/data1/test/003.Birman/Birman_9.jpg', (150, 150)))
# print(predict_img(load_m, './venv/data1/test/004.Bombay/Bombay_3.jpg', (150, 150)))
# print(predict_img(load_m, './venv/data1/test/010.Russian_Blue/Russian_Blue_8.jpg', (150, 150)))
# print('predIdx   ', predIdxs)
