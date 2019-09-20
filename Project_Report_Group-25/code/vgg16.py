from sklearn.datasets import load_files
from keras.utils import np_utils
import numpy as np
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input as preprocess_input_vgg16
from keras import optimizers
from tqdm import tqdm
from PIL import ImageFile
from keras.layers.core import Dropout, Activation
from keras.callbacks import ModelCheckpoint
from keras.layers.normalization import BatchNormalization
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Dense
from keras.layers import Input
from keras.layers import GlobalAveragePooling2D
from keras.preprocessing import image


def load_data_set(path):
    # function of loading images from data set

    # loading images from the given path
    image_data = load_files(path)

    # get the file names
    pets_files = np.array(image_data['filenames'])

    # get the 37 pets as categorical targets and return pets files and pets targets
    pets_targets = np_utils.to_categorical(np.array(image_data['target']), 37)
    return pets_files, pets_targets


def images_to_tensor(img_paths):

    tensor_list = []
    for img_path in tqdm(img_paths):

        # loading and converting the Image from RGB to PIL.Image.Image
        img = image.load_img(img_path, target_size=(224, 224))

        # first reshape the img to (224, 224, 3) through 3D tensor
        # and then reshape it into (1, 224, 224, 3) through 4D tensor and add to the tensor list
        tensor_list.append(np.expand_dims(image.img_to_array(img), axis=0))

    # pile up the matrix in vertical way and return it
    return np.vstack(tensor_list)


def vgg16_extractor(file_paths):
    tensors = images_to_tensor(file_paths).astype('float32')
    preprocessed_input = preprocess_input_vgg16(tensors)
    return VGG16(weights='imagenet', include_top=False).predict(preprocessed_input, batch_size=32)


# loading images from the given path as train, validation and test
train_files, train_targets = load_data_set('./venv/data/images/train')
valid_files, valid_targets = load_data_set('./venv/data/images/validation')
test_files, test_targets = load_data_set('./venv/data/images/test')

# extract files from train, validation and test files
ImageFile.LOAD_TRUNCATED_IMAGES = True
train_vgg16 = vgg16_extractor(train_files)
valid_vgg16 = vgg16_extractor(valid_files)
test_vgg16 = vgg16_extractor(test_files)

# list all pets' breeds
dict_breed = {1: 'Abyssinian', 2: 'American Bulldog', 3: 'American Pit Bull Terrier',
              4: 'Basset Hound', 5: 'Beagle', 6: 'Bengal', 7: 'Birman',
              8: 'Bombay', 9: 'Boxer', 10: 'British Shorthair',
              11: 'Chihuahua', 12: 'Egyptian Mau', 13: 'English Cocker Spaniel', 14: 'English Setter',
              15: 'German Shorthaired', 16: 'Great Pyrenees', 17: 'Havanese', 18: 'Japanese Chin',
              19: 'Keeshond', 20: 'Leonberger', 21: 'Maine Coon', 22: 'Miniature Pinscher',
              23: 'Newfoundland', 24: 'Persian', 25: 'Pomeranian', 26: 'Pug', 27: 'Ragdoll',
              28: 'Russian_Blue', 29: 'Saint Bernard', 30: 'Samoyed', 31: 'Scottish Terrier', 32: 'Shiba Inu',
              33: 'Siamese', 34: 'Sphynx', 35: 'Staffordshire Bull Terrier', 36: 'Wheaten Terrier',
              37: 'Yorkshire Terrier'}

# for check whether loading the data set in a right way
print('There are %d total pets categories.' % len(dict_breed))
print('There are %s total pets images.\n' % len(np.hstack([train_files, valid_files, test_files])))
print('There are %d training pets images.' % len(train_files))
print('There are %d validation pets images.' % len(valid_files))
print('There are %d test pets images.' % len(test_files))

# this is one way to build the vgg_branch


# getting the vgg branch and input with the input shape (7,7,512)
input_vgg16 = Input(shape=(7, 7, 512))
branch = GlobalAveragePooling2D()(input_vgg16)

# initial kernel as random_uniform can get the better performance
# this result is after experimenting
branch = Dense(128, use_bias=False, kernel_initializer='random_uniform')(branch)

# add batchNormalization here
branch = BatchNormalization()(branch)

# using activation function relu
branch_vgg16 = Activation("relu")(branch)
nn = Dropout(0.5)(branch_vgg16)
nn = Dense(512, use_bias=False, kernel_initializer='uniform')(nn)
nn = BatchNormalization()(nn)
nn = Activation("relu")(nn)
nn = Dropout(0.5)(nn)
nn = Dense(37, kernel_initializer='uniform', activation="softmax")(nn)
model = Model(inputs=input_vgg16, outputs=[nn])
model.summary()

# compile the model
model.compile(loss='categorical_crossentropy', optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy', 'top_k_categorical_accuracy'])

# save each model after each epoch into the specific file path
checkpointer = ModelCheckpoint(filepath='saved_models/VGG16_ma.hdf5', verbose=1, save_best_only=True)

# feed the model with the train data and validation data
history = model.fit(train_vgg16, train_targets,
                    validation_data=(valid_vgg16, valid_targets),
                    epochs=30, batch_size=4, callbacks=[checkpointer], verbose=1)

# plot the history for top-1 accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('VGG16 model top-1 accuracy')
plt.ylabel('top-1 accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='lower right')
plt.show()

# plot the history for top-5 accuracy
plt.plot(history.history['top_k_categorical_accuracy'])
plt.plot(history.history['val_top_k_categorical_accuracy'])
plt.title('VGG16 model top-5 accuracy')
plt.ylabel('top-5 accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='lower right')
plt.show()

# plot the history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('VGG16 model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper right')
plt.show()

# loading the weights file and test
model.load_weights('saved_models/VGG16_ma.hdf5')

# make a prediction on the test image sets
predictions = model.predict(test_vgg16)

# get the breed labels and breed prediction results
breed_predictions = [np.argmax(prediction) for prediction in predictions]
breed_true_labels = [np.argmax(true_label) for true_label in test_targets]
print('Test accuracy: %.4f%%' % (accuracy_score(breed_true_labels, breed_predictions) * 100))
