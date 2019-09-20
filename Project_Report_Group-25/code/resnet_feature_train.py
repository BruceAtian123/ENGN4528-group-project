from sklearn.datasets import load_files
from keras.utils import np_utils
import numpy as np
from keras.preprocessing import image
from tqdm import tqdm
from glob import glob
import os
from keras.applications.resnet50 import preprocess_input, decode_predictions, ResNet50
from PIL import ImageFile
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras import applications
from keras.layers import Dropout,Flatten,Dense
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from PIL import Image
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K


# K.set_image_dim_ordering('th')
#
# model = Sequential()
# model.add(Conv2D(32, (3, 3), input_shape=(3, 224, 224)))
# model.add(Activation('relu'))
#
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Conv2D(32, (3, 3)))
# model.add(Activation('relu'))
#
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Conv2D(64, (3, 3)))
# model.add(Activation('relu'))
#
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Flatten()) # this converts our 3D feature maps to 1D feature vectors
# model.add(Dense(64))
# model.add(Activation('relu'))
#
# model.add(Dropout(0.5))
# model.add(Dense(1))
# model.add(Activation('sigmoid'))
#
# model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
# batch_size = 20
#
# # this is the augmentation configuration we will use for training
# train_datagen = ImageDataGenerator(
#                 rescale=1./255,
#                 shear_range=0.2,
#                 zoom_range=0.2,
#                 horizontal_flip=True)
#
# # this is the augmentation configuration we will use for testing: only rescaling
# test_datagen = ImageDataGenerator(rescale=1./255)
#
# # this is a generator that will read pictures found in subfolders of 'data/train', and indefinitely generate
# # batches of augmented image data
# train_generator = train_datagen.flow_from_directory(
#                                     os.getcwd() + '/venv/data1/train',  # this is the target directory
#                                     target_size=(224, 224),  # all images will be resize to 150x150
#                                     batch_size=batch_size,
#                                     class_mode='binary')  # since we use binary_crossentropy loss, we need binary labels
#
# # this is a similar generator, for validation data
# validation_generator = test_datagen.flow_from_directory(
#                                                         os.getcwd() + '/venv/data1/validation',
#                                                         target_size=(224, 224),
#                                                         batch_size=batch_size,
#                                                         class_mode='binary')
# model.fit_generator(
#                     train_generator,
#                     steps_per_epoch=2000 // batch_size,
#                     epochs=1,
#                     validation_data=validation_generator,
#                     validation_steps=800 // batch_size)
#
# model.save_weights('first_try.h5')
# define datagenerator
datagen = ImageDataGenerator(rescale=1./255)
model = applications.VGG16(include_top=False,weights='imagenet')
batch_size = 32

datagen = ImageDataGenerator(
                            rotation_range=40,
                            width_shift_range=0.3,
                            height_shift_range=0.3,
                            shear_range=0.4,
                            zoom_range=0.4,
                            horizontal_flip=True,
                            fill_mode='nearest')

# generate data for train set
generator = datagen.flow_from_directory(
                                        os.getcwd()+'/venv/data1/train',
                                        target_size=(224, 224),
                                        batch_size=batch_size,
                                        class_mode=None,
                                        shuffle=False)
bottleneck_features_train = model.predict_generator(generator, 1196)
np.save('bottleneck_features_train.npy', bottleneck_features_train)

# generate data for validation set
generator = datagen.flow_from_directory(
                                         os.getcwd()+'/venv/data1/validation',
                                         target_size=(224, 224),
                                         batch_size=batch_size,
                                         class_mode=None,
                                         shuffle=False)
bottleneck_features_validation = model.predict_generator(generator, 582)
np.save('bottleneck_features_validation.npy', bottleneck_features_validation)

# generate data for test set
generator = datagen.flow_from_directory(
                                        os.getcwd()+'/venv/data1/test',
                                        target_size=(224, 224),
                                        batch_size=batch_size,
                                        class_mode=None,
                                        shuffle=False)
bottleneck_features_test = model.predict_generator(generator, 593)
np.save('bottleneck_features_test.npy', bottleneck_features_test)


# train_data = np.load(open('bottleneck_features_train.npy'))
# train_labels = np.array([0]*1000+[1]*400)
#
# validation_data = np.load(open('bottleneck_features_validation.npy'))
# validation_labels = np.array([0]*1000+[1]*400)
#
# model.add(Flatten(input_shape=train_data.shape[1:]))
# model.add(Dense(256,activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(1,activation='sigmoid'))
#
# model.compile(optimizer='rmsprop',
#               loss='binary_crossentropy',
#               metrics=['accuracy'])
#
# model.fit(train_data,train_labels,
#           epochs=50,
#           batch_size=batch_size,
#           validation_data=(validation_data,validation_labels))
# model.save_weights('bottleneck_cat_model.h5')
