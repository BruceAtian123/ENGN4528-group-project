import os
import matplotlib.pyplot as plt
import random

# based on the ground truth value to split the images into 3 data sets, train, validation, test
with open('annotations/list.txt') as file:

    # dict and lists for both cats and dogs
    cats = []
    dogs = []
    all_cats = {}
    all_dogs = {}
    cat_name = dict()
    dog_name = dict()

    # read the lines and split it based on '_'
    for lines in file.readlines():
        lines = lines.split(' ')
        name = lines[0].split('_')

        # check whether it is a cat or a dog, 1 for cats, 2 for dogs
        # this is for cats
        if lines[-2] == '1':
            cats.append(lines[0])

            # for short names, with only one '_'
            if len(name) == 2:
                if name[0] in cat_name:
                    cat_name[name[0]] = cat_name[name[0]] + 1
                    all_cats[name[0]] = all_cats[name[0]] + ', ' + lines[0]
                else:
                    cat_name[name[0]] = 1
                    all_cats[name[0]] = lines[0]

            # for long names, with two '_'
            elif len(name) > 2:
                if (str(name[0]) + '_' + str(name[1])) in cat_name:
                    cat_name[str(name[0]) + '_' + str(name[1])] = cat_name[str(name[0]) + '_' + str(name[1])] + 1
                    all_cats[str(name[0]) + '_' + str(name[1])] = all_cats[str(name[0]) + '_' + str(name[1])] + ', ' + lines[0]
                else:
                    cat_name[(str(name[0]) + '_' + str(name[1]))] = 1
                    all_cats[str(name[0]) + '_' + str(name[1])] = lines[0]

        # this is for dogs
        elif lines[-2] == '2':
            dogs.append(lines[0])

            if len(name) == 2:
                if name[0] in dog_name:
                    dog_name[name[0]] = dog_name[name[0]] + 1
                    all_dogs[name[0]] = all_dogs[name[0]] + ', ' + lines[0]
                else:
                    dog_name[name[0]] = 1
                    all_dogs[name[0]] = lines[0]
            elif len(name) > 2:
                if (str(name[0]) + '_' + str(name[1])) in dog_name:
                    dog_name[str(name[0]) + '_' + str(name[1])] = dog_name[str(name[0]) + '_' + str(name[1])] + 1
                    all_dogs[str(name[0]) + '_' + str(name[1])] = all_dogs[str(name[0]) + '_' + str(name[1])] + ', ' + lines[0]
                else:
                    dog_name[str(name[0]) + '_' + str(name[1])] = 1
                    all_dogs[str(name[0]) + '_' + str(name[1])] = lines[0]

# variables for saving cats images and dogs images
cats_image = []
dogs_image = []
cats_keys = all_cats.keys()
dogs_keys = all_dogs.keys()

# split the images into train, validation, test with ratio 2:1:1
for ele in cats_keys:
    all_cats[ele] = all_cats[ele].split(', ')
    for filename in all_cats[ele]:
        img = plt.imread(os.getcwd() + '/' + '/images/' + filename + '.jpg')

        # this lines for the pets shape
        # img = plt.imread(os.getcwd() + '/' + '/annotations/trimaps/' + filename + '.png')

        # use the random value to split the images
        a = random.random()
        if a <= 0.5:
            plt.imsave(os.getcwd() + '/venv/data/train/cats/' + filename + '.jpg', img)
            # plt.imsave(os.getcwd() + '/venv/data/train/cats_shape/' + filename + '.png', img)
        elif a <= 0.75:
            plt.imsave(os.getcwd() + '/venv/data/validation/cats/' + filename + '.jpg', img)
            # plt.imsave(os.getcwd() + '/venv/data/validation/cats_shape/' + filename + '.png', img)
        else:
            plt.imsave(os.getcwd() + '/venv/data/test/cats/' + filename + '.jpg', img)
            # plt.imsave(os.getcwd() + '/venv/data/testing/cats_shape/' + filename + '.png', img)

for ele in dogs_keys:
    all_dogs[ele] = all_dogs[ele].split(', ')
    for filename in all_dogs[ele]:
        img = plt.imread(os.getcwd() + '/' + '/images/' + filename + '.jpg')

        # this lines for the pets shape
        # img = plt.imread(os.getcwd() + '/' + '/annotations/trimaps/' + filename + '.png')

        # use the random value to split the images
        a = random.random()

        if a <= 0.5:
            plt.imsave(os.getcwd() + '/venv/data/train/dogs/' + filename + '.jpg', img)
            # plt.imsave(os.getcwd() + '/venv/data/train/dogs_shape/' + filename + '.png', img)
        elif a <= 0.75:
            plt.imsave(os.getcwd() + '/venv/data/validation/dogs/' + filename + '.jpg', img)
            # plt.imsave(os.getcwd() + '/venv/data/validation/dogs_shape/' + filename + '.png', img)
        else:
            plt.imsave(os.getcwd() + '/venv/data/test/dogs/' + filename + '.jpg', img)
            # plt.imsave(os.getcwd() + '/venv/data/testing/dogs_shape/' + filename + '.png', img)
