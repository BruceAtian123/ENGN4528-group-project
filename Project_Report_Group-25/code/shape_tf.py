from sklearn.datasets import load_files
from keras.utils import np_utils
import numpy as np
from keras.preprocessing import image
from tqdm import tqdm
from glob import glob
import matplotlib.pyplot as plt
import os
from keras.applications.resnet50 import preprocess_input, decode_predictions, ResNet50
from PIL import ImageFile
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import Activation, Dropout, Flatten, Dense
from extract_bottleneck_features import *
from PIL import Image
from keras.layers.normalization import BatchNormalization


# define function to load train, test, and validation datasets
def load_dataset(path):
    data = load_files(path)
    dog_files = np.array(data['filenames'])
    dog_targets = np_utils.to_categorical(np.array(data['target']), 37)
    return dog_files, dog_targets

# define function to load tensor
def to_tensor(img_path):
    # using PIL to transfer the RGB RGb image into PLT.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert the PIL.Image.Image type into 3 dimension like (224, 224, 3)
    x = image.img_to_array(img)
    # convert the 3 D into 4  D like (1, 224, 224, 3)
    return np.expand_dims(x, axis=0)

# define function to predict breed
def ResNet_predict_breed(img_path):
    bottleneck_feature = extract_Resnet50(to_tensor(img_path))
    predicted_vector=ResNet_model.predict(bottleneck_feature)
    return cat_names[np.argmax(predicted_vector)]


def cat(img_path):
    return 'Cat Breed: ' + str(img_path)


def convert_to_tensor(img_paths):
    tensors = []
    for img_path in tqdm(img_paths):
        tensor = to_tensor((img_path))
        tensors.append(tensor)
        tensors = np.vstack(tensors)
    return tensors

def ResNet50_predict_labels(img_path):
    img = preprocess_input(to_tensor(img_path))
    return np.argmax(ResNet50_model.predict(img))

# Predict breed by using vgg16 model
def VGG16_predict_breed(img_path):
    # extract bottleneck features
    bottleneck_feature = extract_VGG16(to_tensor(img_path))
    # obtain predicted vector
    predicted_vector = VGG16_model.predict(bottleneck_feature)
    # return dog breed that is predicted by the model
    return cat_names[np.argmax(predicted_vector)]

# define resnet50 model by giving pretrained weight
ResNet50_model = ResNet50(weights='imagenet')
ImageFile.LOAD_TRUNCATED_IMAGES = True

# load train, test, and validation datasets
train_files, train_targets = load_dataset(os.getcwd() + '/venv/data1/train')
valid_files, valid_targets = load_dataset(os.getcwd() + '/venv/data1/validation')
test_files, test_targets = load_dataset(os.getcwd() + '/venv/data1/test')

train_tensors = convert_to_tensor(train_files).astype('float32') / 255
valid_tensors = convert_to_tensor(valid_files).astype('float32') / 255
test_tensors = convert_to_tensor(test_files).astype('float32') / 255

# load list of dog names

dog_name = set()
with open('annotations/list.txt') as file:
    for lines in file.readlines():
        split_line = lines.split(' ')
        name = split_line[0]
        name = name.split('_')

        if name[0][0] == '#':
            continue
        else:
            if len(name) > 2:
                dog_name.add(name[0] + '_' + name[1])
            else:
                dog_name.add(name[0])

dog_name = list(dog_name)

model = Sequential()

model.add(Conv2D(filters=32, kernel_size=3, padding='valid', activation='relu',input_shape=(224, 224, 3)))
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters=64, kernel_size=3, padding='valid', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.25))

model.add(Conv2D(filters=256, kernel_size=3, padding='valid', activation='relu'))
model.add(MaxPooling2D(pool_size=2))
model.add(Dropout(0.5))

model.add(GlobalAveragePooling2D(data_format='channels_last'))
model.add(Dense(37, activation='softmax'))

model.summary()

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

epochs = 1

checkpointer = ModelCheckpoint(filepath=os.getcwd()+'/saved_models/weights.best.from_scratch.hdf5',
                               verbose=1, save_best_only=True)
model.fit(train_tensors, train_targets,
          validation_data=(valid_tensors, valid_targets),
          epochs=epochs, batch_size=50, callbacks=[checkpointer], verbose=1)

model.load_weights(os.getcwd()+'/saved_models/weights.best.from_scratch.hdf5')

cat_breed_predictions = [np.argmax(model.predict(np.expand_dims(tensor, axis=0))) for tensor in test_tensors]

test_accuracy = 100*np.sum(np.array(cat_breed_predictions)==np.argmax(test_targets, axis=1))/len(cat_breed_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)

train_VGG16 = np.load(os.getcwd()+'/train.npy')
valid_VGG16 = np.load(os.getcwd()+'/valid.npy')
test_VGG16 = np.load(os.getcwd()+'/test.npy')

VGG16_model = Sequential()
VGG16_model.add(GlobalAveragePooling2D(input_shape=train_VGG16.shape[1:]))
VGG16_model.add(Dense(37, activation='softmax'))

VGG16_model.summary()

VGG16_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

checkpointer = ModelCheckpoint(filepath=os.getcwd()+'/saved_models/weights.best.VGG16.hdf5',
                               verbose=1, save_best_only=True)

VGG16_model.fit(train_VGG16, train_targets,
                validation_data=(valid_VGG16, valid_targets),
                epochs=500, batch_size=20, callbacks=[checkpointer], verbose=1)

VGG16_model.load_weights(os.getcwd()+'/saved_models/weights.best.VGG16.hdf5')

# get index of predicted dog breed for each image in test set
VGG16_predictions = [np.argmax(VGG16_model.predict(np.expand_dims(feature, axis=0))) for feature in test_VGG16]

# report test accuracy
test_accuracy = 100 * np.sum(np.array(VGG16_predictions) == np.argmax(test_targets, axis=1))/len(VGG16_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)

# using CNN to classify the cat breed
train_ResNet = np.load(os.getcwd()+'/train.npy')
valid_ResNet = np.load(os.getcwd()+'/valid.npy')
test_ResNet = np.load(os.getcwd()+'/test.npy')

ResNet_model = Sequential()
ResNet_model.add(GlobalAveragePooling2D(input_shape=train_ResNet.shape[1:]))
ResNet_model.add(Dense(37, activation='softmax'))
ResNet_model.summary()

ResNet_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy', 'top_k_categorical_accuracy'])

checkpointer_r = ModelCheckpoint(filepath=os.getcwd()+'/saved_models/weights.best.ResNet_model.hdf5',
                                 verbose=1, save_best_only=True)

ResNet_model.fit(train_ResNet, train_targets,
                 validation_data=(valid_ResNet, valid_targets),
                 epochs=50, batch_size=16, callbacks=[checkpointer_r], verbose=1)

ResNet_model.load_weights(os.getcwd()+'/saved_models/weights.best.ResNet_model.hdf5')

# get index of predicted dog breed for each image in test set
ResNet_predictions = [np.argmax(ResNet_model.predict(np.expand_dims(feature, axis=0))) for feature in test_ResNet]

# report test accuracy
test_accuracy = 100 * np.sum(np.array(ResNet_predictions) == np.argmax(test_targets, axis=1))/len(ResNet_predictions)
print('Test accuracy: %.4f%%' % test_accuracy)

cat_names = [item[20:] for item in sorted(glob(os.getcwd() + '/venv/data/images/test/*'))]
cat_files = np.array(glob((os.getcwd()+'/venv/data/images/test/*')))
cats_name = [i[48:] for i in cat_names]

# predict breed here
for i in range(len(cat_names)):
    im = Image.open(cat_files[i])
    plt.imshow(im)
    plt.show()
    print('File Name: ' + cat_names[i])
    print(cat(cat_files[i]))
    print('\n')
