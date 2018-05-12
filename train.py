"""
Adapted from keras example cifar10_cnn.py
Train ResNet-18 on the CIFAR10 small images dataset.

GPU run command with Theano backend (with TensorFlow, the GPU is automatically used):
	THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python cifar10.py
"""
from __future__ import print_function
import os
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping, TensorBoard, ModelCheckpoint

import sys
import numpy as np
import resnet
import pickle
import exp_config as exp_config
from fcns.imshow3d import imshow3d
from keras import optimizors

def progressBar(value, endvalue, bar_length=20):
	percent = float(value) / endvalue
	arrow = '-' * int(round(percent * bar_length) - 1) + '>'
	spaces = ' ' * (bar_length - len(arrow))
	sys.stdout.write(
		"\r{0} of {1} Percent: [{2}] {3}%".format(value, endvalue, arrow + spaces, round(percent * 100000) / 1000))
	sys.stdout.flush()

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
early_stopper = EarlyStopping(min_delta=0.001, patience=20)
csv_logger = CSVLogger('resnet18_cifar10.csv')

batch_size = 16
nb_classes = 2
nb_epoch = 500
data_augmentation = True

# input image dimensions
img_rows, img_cols = 160, 160
# The CIFAR10 images are RGB.
img_channels = 7

# The data, shuffled and split between train and test sets:
# (X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Datasets
with open('C://Users\qiangz\Documents//180101_HCMR_artefacts\gradcam_scoring_active\data\motion//train/partition_labels.pkl', 'rb') as f:
	print('loading partition and labels from [partition_labels.pkl] ... ')
	partition, labels = pickle.load(f)
	print('loaded')
# load validation set
partion_val = partition['validation']
nvalid = len(partion_val)
X_valid = np.empty((nvalid, exp_config.image_size[0], exp_config.image_size[1], exp_config.image_depth, 1))
for k, file in zip(range(nvalid), partion_val):
	progressBar(k + 1, nvalid)
	X_valid[k, :, :, :, 0] = np.load(exp_config.datapath + 'train//image//' + file + '.npy')  # [:,:,1:]

y_valid = [labels[key] for key in partion_val]
Y_valid = np.array([[1 if y_valid[i] == j else 0 for j in range(2)] for i in range(len(y_valid))])

partition_chunk = partition['train']
ntrain = len(partition_chunk)
X_train = np.empty((ntrain, exp_config.image_size[0], exp_config.image_size[1], exp_config.image_depth, 1))
for k, file in zip(range(ntrain), partition_chunk):
	progressBar(k + 1, ntrain)
	X_train[k, :, :, :, 0] = np.load(exp_config.datapath + 'train//image//' + file + '.npy')  # [:,:,1:]

y_train = [labels[key] for key in partition_chunk]


Y_train = np.array([[1 if y_train[i] == j else 0 for j in range(2)] for i in range(len(y_train))])




# for k in range(len(Y_train)):
#     if Y_train[k][0] == 0:
#         X_train[k,:,:,:,0] = np.flip(X_train[k,:,:,:,0],axis=1)
#
# for k in range(len(Y_test)):
#     if Y_test[k][0] == 0:
#         X_valid[k, :, :, :, 0] = np.flip(X_valid[k, :, :, :, 0], axis=1)

# # Convert class vectors to binary class matrices.
# Y_train = np_utils.to_categorical(y_train, nb_classes)
# Y_test = np_utils.to_categorical(y_test, nb_classes)

for k in range(10):
	if y_train[k] is 1:
		imshow3d(X_train[k, :, :, :, 0],title='positive')

	if y_train[k] is 0:
		imshow3d(X_train[k, :, :, :, 0], title='negative')





X_train = X_train.astype('float32')
X_valid = X_valid.astype('float32')
X_train = X_train.squeeze()
X_valid = X_valid.squeeze()

# subtract mean and normalize
# mean_image = np.mean(X_train, axis=0)
# X_train -= mean_image
# X_test -= mean_image
# X_train /= 128.
# X_test /= 128.

model = resnet.ResnetBuilder.build_resnet_18((img_channels, img_rows, img_cols), nb_classes)
opt = optimizers.adam(lr=0.005)
model.compile(loss='categorical_crossentropy',
			  optimizer=opt,
			  metrics=['accuracy'])

filepath = exp_config.datapath + "weights_resnet//{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

if not data_augmentation:
	print('Not using data augmentation.')
	model.fit(X_train, Y_train,
			  batch_size=batch_size,
			  nb_epoch=nb_epoch,
			  validation_data=(X_valid, Y_valid),
			  shuffle=True,
			  callbacks=[lr_reducer, early_stopper, csv_logger])
else:
	print('Using real-time data augmentation.')
	# This will do preprocessing and realtime data augmentation:
	datagen = ImageDataGenerator(
		featurewise_center=False,  # set input mean to 0 over the dataset
		samplewise_center=False,  # set each sample mean to 0
		featurewise_std_normalization=False,  # divide inputs by std of the dataset
		samplewise_std_normalization=False,  # divide each input by its std
		zca_whitening=False,  # apply ZCA whitening
		rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
		width_shift_range=0,  # randomly shift images horizontally (fraction of total width)
		height_shift_range=0,  # randomly shift images vertically (fraction of total height)
		shear_range=0,
		horizontal_flip=False,  # randomly flip images
		vertical_flip=False)  # randomly flip images

	# Compute quantities required for featurewise normalization
	# (std, mean, and principal components if ZCA whitening is applied).
	# datagen.fit(X_train)
	tbCallBack = TensorBoard(log_dir='./Graph')


	# Fit the model on the batches generated by datagen.flow().
	model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size),
						# steps_per_epoch=X_train.shape[0] // batch_size,
						validation_data=datagen.flow(X_valid, Y_valid,batch_size=batch_size),
						# validation_steps=len(partion_val) / batch_size,
						epochs=nb_epoch, verbose=1, max_q_size=100,
						callbacks=[lr_reducer, csv_logger,checkpoint, tbCallBack])



