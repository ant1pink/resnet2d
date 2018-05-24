"""
Adapted from keras example cifar10_cnn.py
Train ResNet-18 on the CIFAR10 small images dataset.
GPU run command with Theano backend (with TensorFlow, the GPU is automatically used):
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python cifar10.py
"""
from __future__ import print_function
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator, img_to_array
from keras.preprocessing.image import load_img
from keras.utils import np_utils
from keras.callbacks import TensorBoard
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping
import os
import numpy as np
import resnet
import cv2
from sklearn import preprocessing
from keras.applications.resnet50 import preprocess_input
from keras.callbacks import ModelCheckpoint

#lr_reducer = ReduceLROnPlateau(monitor = 'val_loss' factor=np.sqrt(0.3), cooldown=0, patience=5, min_lr=0.5e-6)
early_stopper = EarlyStopping(min_delta=0.001, patience=10)
csv_logger = CSVLogger('resnet18_cifar10.csv')

batch_size = 16
nb_classes = 23
nb_epoch = 150
data_augmentation = False


# input image dimensions
img_rows, img_cols = 224, 224
# The CIFAR10 images are RGB.
img_channels = 3


# def files_from_dir(dir):
#     # initialize the data and labels
#     print("[INFO] loading images...")
#     data = []
#     labels = []
#     imagePaths = []
#     for forlder in os.listdir(dir):
#         for file in os.listdir(os.path.join(dir,forlder)):
#             imagePaths.append(os.path.join(dir, forlder, file))
#     #
#     # random.seed(1)
#     # random.shuffle(imagePaths)
#
#     # loop over the input images
#     for imagePath in imagePaths:
#         # load the image, pre-process it, and store it in the data list
#         image = load_img(imagePath, target_size=(img_rows, img_cols))
#         # image = cv2.imread(imagePath)
#         # image = cv2.resize(image, (img_rows, img_cols))
#         # cv2.imshow('Tracking',image)
#         # un-comment for grayscale
#         # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         image = img_to_array(image)
#         data.append(image)
#         # extract the class label from the image path and update the labels list
#         label = imagePath.split(os.path.sep)[-2]
#         # TODO:convert to categorical
#         # append 0 for autoencoder
#         labels.append(label)
#     le = preprocessing.LabelEncoder()
#     labels = le.fit_transform(labels)
#     labels = np_utils.to_categorical(labels, nb_classes)
#     # labels = np.array(labels)
#     data = np.array(data, dtype="float32")
#     # data = data.astype('float32')
#     return train_test_split(data, labels, test_size=0.1, random_state=1)
#
# X_train, X_test, Y_train, Y_test = files_from_dir('C:/Users/ruili2.LL/Desktop/img_new/')
#
# def preprocess_data(x):
#     # subtract mean and normalize
#     x /= 127.5
#     x -= 1.
#     return x
#
# X_train = preprocess_data(X_train)
# X_test = preprocess_data(X_test)
#
# print(X_train.shape, Y_train.shape)

# # The data, shuffled and split between train and test sets:
# (X_train, y_train), (X_test, y_test) = cifar10.load_data()


from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(samplewise_center=True, samplewise_std_normalization = True, validation_split = 0.2)
train_generator = train_datagen.flow_from_directory('img_new/', target_size= (img_rows, img_cols), batch_size=32, class_mode = 'categorical', subset ='training')
valid_generator = train_datagen.flow_from_directory('img_new/', target_size= (img_rows, img_cols), batch_size=32, class_mode = 'categorical', subset ='validation')

# # Convert class vectors to binary class matrices.
# Y_train = np_utils.to_categorical(y_train, nb_classes)
# Y_test = np_utils.to_categorical(y_test, nb_classes)
#
# X_train = X_train.astype('float32')
# X_test = X_test.astype('float32')
#
# # subtract mean and normalize
# mean_image = np.mean(X_train, axis=0)
# X_train -= mean_image
# X_test -= mean_image
# X_train /= 128.
# X_test /= 128.

model = resnet.ResnetBuilder.build_resnet_18((img_channels, img_rows, img_cols), nb_classes)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

tbCallBack = TensorBoard(log_dir='./Graph')

# if not data_augmentation:
#     print('Not using data augmentation.')
#     model.fit(X_train, Y_train,
#               batch_size=batch_size,
#               nb_epoch=nb_epoch,
#               validation_data=(X_test, Y_test),
#               shuffle=True,
#               callbacks=[lr_reducer, early_stopper, csv_logger, tbCallBack])
# else:
#     print('Using real-time data augmentation.')
#     # This will do preprocessing and realtime data augmentation:
#     datagen = ImageDataGenerator(
#         featurewise_center=False,  # set input mean to 0 over the dataset
#         samplewise_center=False,  # set each sample mean to 0
#         featurewise_std_normalization=False,  # divide inputs by std of the dataset
#         samplewise_std_normalization=False,  # divide each input by its std
#         zca_whitening=False,  # apply ZCA whitening
#         rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
#         width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
#         height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
#         horizontal_flip=True,  # randomly flip images
#         vertical_flip=False)  # randomly flip images
#
#     # Compute quantities required for featurewise normalization
#     # (std, mean, and principal components if ZCA whitening is applied).
#     datagen.fit(X_train)
#
#
#     # Fit the model on the batches generated by datagen.flow().
#     model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),
#                         steps_per_epoch=X_train.shape[0] // batch_size,
#                         validation_data=(X_test, Y_test),
#                         epochs=nb_epoch, verbose=1, max_q_size=100,
#                         callbacks=[lr_reducer, early_stopper, csv_logger, tbCallBack])
checkpointer = ModelCheckpoint(filepath='./weights/resnet18_weights.hdf5', verbose=1, save_best_only=True)
model.fit_generator(train_generator,
                    # steps_per_epoch=X_train.shape[0] // batch_size,
                    steps_per_epoch = train_generator.samples // train_generator.batch_size,
                    validation_data = valid_generator,
                    epochs=nb_epoch, verbose=2, max_queue_size=100,
                    callbacks=[checkpointer, tbCallBack])
