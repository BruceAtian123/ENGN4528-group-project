# import modules
import itertools
from keras.metrics import top_k_categorical_accuracy
from keras.preprocessing.image import ImageDataGenerator
from sklearn.datasets import load_files
from keras.utils import to_categorical
from keras import metrics
import numpy as np
from keras import optimizers
import matplotlib.pyplot as plt
from keras.preprocessing import image
from tqdm import tqdm
from PIL import ImageFile
from keras.applications.xception import Xception
from keras.applications.xception import preprocess_input,decode_predictions
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers import Input, Dense
from keras.layers.core import Dropout, Activation
from keras.callbacks import ModelCheckpoint
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from sklearn.metrics import accuracy_score
import cv2
import time
from keras.regularizers import l2

# useful constants
NUM_CLASSES = 37
IMG_SIZE = 224
BATCH_SIZE = 32


def dataset_loader(path):
    data = load_files(path)
    image_files = np.array(data['filenames'])
    image_targets = to_categorical(np.array(data['target']), NUM_CLASSES)
    return image_files, image_targets

# load the dataset and label for each image

train_images, train_labels = dataset_loader('data/images/train')
valid_images, valid_labels = dataset_loader('data/images/validation')
test_images, test_labels = dataset_loader('data/images/test')

num_train_samples=len(train_images)
num_valid_samples=len(valid_images)
num_test_samples=len(test_images)
num_train_samples=len(train_images)
num_valid_samples=len(valid_images)
num_test_samples=len(test_images)

# general statistical information about the dataset
print('%d categories in total.' % NUM_CLASSES)
print('%s  images in total.' % str(num_train_samples+num_test_samples+num_valid_samples))
print('%d training images.' % num_train_samples)
print('%d validation images.' % num_valid_samples)
print('%d test images.'% num_test_samples)
print('\n')
print('\n')
print('\n')

all_images = list(itertools.chain(train_images, valid_images, test_images))
heights = []
widths = []
for img_path in all_images:
    img = image.load_img(img_path)
    wid, hei = img.size
    heights.append(hei)
    widths.append(wid)
avg_height = sum(heights) / len(heights)
avg_width = sum(widths) / len(widths)
print("average height of all images: " + str(avg_height))
print("max height: " + str(max(heights)))
print("min height: " + str(min(heights)))
print("average width of all images: " + str(avg_width))
print("max width: " + str(max(widths)))
print("min width: " + str(min(widths)))
print('\n')
print('\n')
print('\n')


def convert_to_tensors(img_paths):
    tensors = []
    for img_path in tqdm(img_paths):
        # convert PIL.JpegImagePlugin.JpegImageFile to 3D tensor with the shape (224, 224, 3)
        img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
        # convert PIL image to a numpy array
        numpy_image = image.img_to_array(img)
        # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3)
        tensor = np.expand_dims(numpy_image, axis=0)
        tensors.append(tensor)
        tensors = np.vstack(tensors)
    return tensors

ImageFile.LOAD_TRUNCATED_IMAGES = True

# dict_breed_dogs={0:'American Bulldog',1:'American Pit Bull Terrier',2:'Basset Hound',3:'Beagle',
#                  4:'Boxer',5:'Chihuahua',6:'English Cocker Spaniel',7:'English Setter',
#                  8:'German Shorthaired',9:'Great Pyrenees',10:'Havanese',
#                  11:'Japanese chin', 12:'Keeshond',13:'Leonberger',14:'Miniature Pinscher',
#                  15:'Newfindland',16:'Pomeranian',17:'Pug',18:'Saint Berbard',
#                  19:'Samoyed',20:'Scottish Terrier',21:'Shiba Inu',22:'Staffordshire Bull Terrier',
#                  23:'Wheaten Terrier',24:'Yorkshire Terrier'}

dict_breed={0:'Abyssinian',1:'American Bulldog',2:'American Pit Bull Terrier',3:'Basset Hound',4:'Beagle',
            5:'Bengal',6:'Birman',7:'Bombay',8:'Boxer',9:'British Shorthair',10:'Chihuahua',11:'Egyptian Mau',
            12:'English Cocker Spaniel',13:'English Setter',14:'German Shorthaired',15:'Great Pyrenees',
            16:'Havanese',17:'Japanese chin',18:'Keeshond',19:'Leonberger',20:'Maine Coon',
            21:'Miniature Pinscher',22:'Newfindland',23:'Persian',24:'Pomeranian',
            25:'Pug',26:'Ragdoll',27:'Russian Blue',28:'Saint Berbard',29:'Samoyed',30:'Scottish Terrier',
            31:'Shiba Inu',32:'Siamese',33:'Sphynx',
            34:'Staffordshire Bull Terrier',35:'Wheaten Terrier',36:'Yorkshire Terrier',
           }

start = time.time()

#Load the Xception model
base_Xception_model =Xception(weights='imagenet', include_top=False)
def extract_Xception(file_paths):
    tensors = convert_to_tensors(file_paths).astype('float32')
    preprocessed_input = preprocess_input(tensors)
    return base_Xception_model.predict(preprocessed_input, batch_size=32)

# Extract bottleneck features
train_Xception = extract_Xception(train_images)
valid_Xception = extract_Xception(valid_images)
test_Xception = extract_Xception(test_images)
print("Xception shape: ", train_Xception.shape[1:])

# Generate input and branch for Xception model
input_shape=train_Xception.shape[1:]
size = int(input_shape[2] / 4)
xception_input = Input(shape=input_shape)
xception_branch = GlobalAveragePooling2D()(xception_input)
xception_branch= Dense(size, use_bias=False, kernel_initializer='uniform')(xception_branch)
xception_branch = BatchNormalization()(xception_branch)
xception_branch = Activation("relu")(xception_branch)

# Build the net
net = Dropout(0.5)(xception_branch)
net = Dense(640, use_bias=False, kernel_initializer='uniform',kernel_regularizer=l2(0.01))(net)
net = BatchNormalization()(net)
net = Activation("relu")(net)
net = Dropout(0.5)(net)
net = Dense(37, kernel_initializer='uniform', activation="softmax",kernel_regularizer=l2(0.01))(net)

model = Model(inputs=xception_input, outputs= net)
model.summary()

model.compile(loss='categorical_crossentropy', optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy','top_k_categorical_accuracy'])




checkpointer = ModelCheckpoint(filepath='saved_models/xcep_bestmodel.hdf5',
                               verbose=1, save_best_only=True)
history = model.fit(train_Xception, train_labels,
          validation_data=(valid_Xception, valid_labels),
          epochs=30, batch_size=4, callbacks=[checkpointer], verbose=1)
end = time.time()

model.load_weights('saved_models/xcep_bestmodel.hdf5')

predictions = model.predict(test_Xception)


# plot the history for top-1 accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Xception model top-1 accuracy')
plt.ylabel('top-1 accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

# plot the history for top-5 accuracy
plt.plot(history.history['top_k_categorical_accuracy'])
plt.plot(history.history['val_top_k_categorical_accuracy'])
plt.title('Xception model top-5 accuracy')
plt.ylabel('top-5 accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

# plot the history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Xception model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

# calculate metrics for test set
predictions = [np.argmax(prediction) for prediction in predictions]
truth = [np.argmax(label) for label in test_labels]
print('Running Time: ', end - start)
print('Test Top-1 Accuracy: %.6f%%' % (accuracy_score(truth, predictions) * 100))
print('Test Top-5 Accuracy: %.6f%%' % (top_k_categorical_accuracy(truth, predictions) * 100))

Xception_model = Xception(weights='imagenet')
# Judge the image is dog or cat by ImageNet100 dictionary keys
def is_dog(img_path):
    img = image.load_img(img_path, target_size=(299, 299))
    x = image.img_to_array(img)
    im = preprocess_input(np.expand_dims(x, axis=0))
    prediction = np.argmax(Xception_model.predict(im))
    if (prediction <= 268) & (prediction >= 151):# these classes are dogs in imagenet1000
        return True
    else:
        return False

# predict the pet breed by using the above model
def predict_breed(img_path):
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    x = image.img_to_array(img)
    tensor = np.expand_dims(x, axis=0).astype('float32')
    preprocessed_input = preprocess_input(tensor)
    extracted_input = base_Xception_model.predict(preprocessed_input, batch_size=32)
    predicted_vector = model.predict(extracted_input)
    return np.argmax(predicted_vector)

# print out the results
def show_prediction(img_path):
    img = cv2.imread(img_path)
    cv_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(cv_rgb)
    if is_dog(img_path):
        print("dog OR cat: DOG")
        key = int(predict_breed(img_path))
        print("predicted breed: "+ dict_breed.get(key))
    else:
        print("dog OR cat: CAT")
        key = int(predict_breed(img_path))
        print("predicted breed: "+ dict_breed.get(key))

# Test with the images from outsides
show_prediction('data/presentation_images/pug.jpeg')
show_prediction('data/presentation_images/chi.jpg')
show_prediction('data/presentation_images/samoyed.jpg')
show_prediction('data/presentation_images/hava.jpeg')
show_prediction('data/presentation_images/leon.jpg')
show_prediction('data/presentation_images/eng_sho.jpeg')
show_prediction('data/presentation_images/aby.jpg')
show_prediction('data/presentation_images/egy.jpeg')
show_prediction('data/presentation_images/mini.jpeg')