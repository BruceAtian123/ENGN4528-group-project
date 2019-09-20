1. data_preprocessing.py is used to split the images into train, validation and test sets based on the ground truth value with the ratio 2:1:1. Just run it directly.
2. cnn_scratch.py is the basic cnn model from the scratch. To train and test the model, just run it directly.
3. Run the resnet_feature.py at first to extract resnet50 bottleneck features, and then run shape_tf.py to train, validate and test the model.
4. vgg16.py contains complete code for vgg16 architecture.
5. Run xception.py to extract bottleneck features based on Xception, and then train and test the model. This is the best-performance model, so we use it as the classifer to generate the predictions for external dog or cat images.
