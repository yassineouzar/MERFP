import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import tensorflow as tf
#tf.config.set_soft_device_placement(True)
#tf.debugging.set_log_device_placement(True)

# Disable eager execution (Adam optimizer cannot be used if this option is enabled)
tf.compat.v1.disable_eager_execution()
tf.executing_eagerly()
import numpy as np
import statistics
from numpy import *
import matplotlib.pyplot as plt
import csv
from PIL import Image
from sklearn import preprocessing
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img
from tensorflow.keras.preprocessing.sequence import *
from collections import defaultdict
from collections import Counter

import keras
from keras import backend as K
from keras.models import Model

from keras.models import Sequential
from keras.layers import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.layers.core import Activation, Flatten, Dense, Dropout
#from keras.layers import Activation, Dense
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D, Conv3D, AveragePooling2D
from keras.layers.pooling import MaxPooling3D, AveragePooling3D, GlobalAveragePooling3D
from keras.optimizers import SGD, Adam
from keras.preprocessing.image import ImageDataGenerator

from keras.activations import relu
import cv2
from tensorflow.python.keras.utils import np_utils
from keras import backend as K
#K.common.image_dim_ordering()
#K.set_image_dim_ordering('th')
from keras.utils.vis_utils import plot_model
from keras.layers import Conv3D, ConvLSTM2D, Conv3DTranspose,Input
from keras import optimizers, losses
from keras.layers import *
from keras.models import Model
from keras.backend import int_shape
from keras.layers import Dense
from keras_radam import RAdam
from keras.layers import Dense
from keras.regularizers import l2, l1, l1_l2
#os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
from keras.callbacks import ModelCheckpoint
#from imageGenerator1 import ImageDataGenerator
#from imagegen_csv import ImageDataGenerator
from Generator_yuv1 import ImageDataGenerator
datagen1 = ImageDataGenerator()

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
batch_size=31
#datagen = ImageDataGenerator()
train_data = datagen1.flow_from_directory('E:/BP4D/ROI_BP4D_save', 'E:/BP4D/HR_BP4D_save', (120, 160), class_mode='label', batch_size=1, frames_per_step=50, shuffle=False)

#train_data = datagen.flow_from_directory('D:/HR_estimation/ROI1', 'D:/HR_estimation/heart_rate', target_size=(120, 120), class_mode='label', batch_size=1, frames_per_step=75, shuffle=False)
#test_data=datagen.flow_from_directory('D:/HR_estimation/ROI2', 'D:/HR_estimation/heart_rate2', target_size=(120, 120), class_mode='label', batch_size=1, frames_per_step=75)train_data = datagen1.flow_from_directory('E:/BP4D/ROI_BP4D_save', 'E:/BP4D/HR_BP4D_save', (120, 160), class_mode='label', batch_size=batch_size, frames_per_step=50, shuffle=False)
test_data = datagen1.flow_from_directory(directory='F:/prediction/ROI', label_dir='F:/prediction/HR', target_size=(120, 160), class_mode='label', batch_size=9, frames_per_step=50, shuffle=False)
print("finished1")
from keras.models import Model
from keras import layers
from keras.layers import Dense, Input, BatchNormalization, Activation
from keras.layers import Conv3D, MaxPooling3D, GlobalAveragePooling3D, GlobalMaxPooling2D
#from keras.applications.imagenet_utils import _obtain_input_shape
from keras.utils.data_utils import get_file
import keras
from DepthwiseConv3D import DepthwiseConv3D
#WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.4/xception_weights_tf_dim_ordering_tf_kernels.h5'
SeparableConv3D = DepthwiseConv3D

def se_block(block_input, num_filters, ratio=8):  # Squeeze and excitation block

    '''
        Args:
            block_input: input tensor to the squeeze and excitation block
            num_filters: no. of filters/channels in block_input
            ratio: a hyperparameter that denotes the ratio by which no. of channels will be reduced

        Returns:
            scale: scaled tensor after getting multiplied by new channel weights
    '''

    pool1 = GlobalAveragePooling3D()(block_input)
    flat = Reshape((1, 1, 1, num_filters))(pool1)
    dense1 = Dense(num_filters // ratio, activation='relu')(flat)
    dense2 = Dense(num_filters, activation='sigmoid')(dense1)
    scale = multiply([block_input, dense2])

    return scale

def Xception():

	# Determine proper input shape
	#input_shape = _obtain_input_shape(None, default_size=299, min_size=71, data_format='channels_last', include_top=False)


    img_input = keras.layers.Input(shape=(50, 120, 160, 3))

	# Block 1
    x = Conv3D(32, (3, 3, 3), strides=(2, 2, 2), kernel_regularizer=l1_l2(l1=0.001, l2=0.001), use_bias=False)(img_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv3D(64, (3, 3, 3), kernel_regularizer=l1_l2(l1=0.001, l2=0.001), use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    residual = Conv3D(32, (1, 1, 1), strides=(2, 2, 2), padding='same',kernel_regularizer=l1_l2(l1=0.001, l2=0.001), use_bias=False)(x)

    residual = BatchNormalization()(residual)

    se = se_block(residual, num_filters=32)

    sum = Add()([residual, se])
    relu2 = Activation('relu')(sum)

    # Block 2
    x = SeparableConv3D(kernel_size=(3, 3, 3), depth_multiplier=1)(relu2)

    # x = SeparableConv3D(128, (3, 3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # x = SeparableConv3D(128, (3, 3, 3), padding='same', use_bias=False)(x)
    x = SeparableConv3D(kernel_size=(3, 3, 3), depth_multiplier=1)(x)
    x = BatchNormalization()(x)

    # Block 2 Pool
    x = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same')(x)
    x = layers.add([x, residual])

    residual = Conv3D(64, (1, 1, 1), strides=(2, 2, 2), padding='same',kernel_regularizer=l1_l2(l1=0.001, l2=0.001), use_bias=False)(x)

    residual = BatchNormalization()(residual)
    se = se_block(residual, num_filters=64)

    sum = Add()([residual, se])
    relu3 = Activation('relu')(sum)

    # Block 3
    x = SeparableConv3D(kernel_size=(3, 3, 3), depth_multiplier=1)(relu3)

    # x = SeparableConv3D(256, (3, 3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv3D(kernel_size=(3, 3, 3), depth_multiplier=1)(x)

    # x = SeparableConv3D(256, (3, 3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)

    # Block 3 Pool
    x = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same')(x)

    x = layers.add([x, residual])

    residual = Conv3D(64, (1, 1, 1), strides=(2, 2, 2), padding='same',kernel_regularizer=l1_l2(l1=0.001, l2=0.001),use_bias=False)(x)

    residual = BatchNormalization()(residual)
    se = se_block(residual, num_filters=64)

    sum = Add()([residual, se])
    relu4 = Activation('relu')(sum)

    # Block 4
    x = SeparableConv3D(kernel_size=(3, 3, 3), depth_multiplier=1)(relu4)

    # x = SeparableConv3D(728, (3, 3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv3D(kernel_size=(3, 3, 3), depth_multiplier=1)(x)

    # x = SeparableConv3D(768, (3, 3, 3), padding='same', use_bias=False)(x)

    x = BatchNormalization()(x)

    x = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same')(x)

    x = layers.add([x, residual])


    # Block 5 - 12
    for i in range(8):
        residual=x

        x = Activation('relu')(x)
        x = SeparableConv3D(kernel_size=(3, 3, 3), depth_multiplier=1)(x)

        #x = SeparableConv3D(728, (3, 3, 3), padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = SeparableConv3D(kernel_size=(3, 3, 3), depth_multiplier=1)(x)

        #x = SeparableConv3D(728, (3, 3, 3), padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = SeparableConv3D(kernel_size=(3, 3, 3), depth_multiplier=1)(x)

        #x = SeparableConv3D(728, (3, 3, 3), padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = layers.add([x, residual])

    residual = Conv3D(64, (1, 1, 1), strides=(2, 2, 2), padding='same',kernel_regularizer=l1_l2(l1=0.001, l2=0.001),use_bias=False)(x)

    residual = BatchNormalization()(residual)
    se = se_block(residual, num_filters=64)

    sum = Add()([residual, se])
    relu5 = Activation('relu')(sum)

    # Block 13
    x = SeparableConv3D(kernel_size=(3, 3, 3), depth_multiplier=1)(relu5)

    # x = SeparableConv3D(728, (3, 3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv3D(kernel_size=(3, 3, 3), depth_multiplier=1)(x)

    # x = SeparableConv3D(1024, (3, 3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)

    # Block 13 Pool
    x = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same')(x)

    x = layers.add([x, residual])

    # Block 14
    x = SeparableConv3D(kernel_size=(3, 3, 3), depth_multiplier=1)(x)

    # x = SeparableConv3D(1536, (3, 3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Block 14 part 2
    x = SeparableConv3D(kernel_size=(3, 3, 3), depth_multiplier=2)(x)

    # x = SeparableConv3D(2048, (3, 3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    se = se_block(x, num_filters=128)

    sum = Add()([x, se])
    relu6 = Activation('relu')(sum)

    # Fully Connected Layer
    # x = GlobalAveragePooling3D()(x)

    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(1024, kernel_regularizer=l1_l2(l1=0.001, l2=0.001), activation='relu')(x)
    #x = keras.layers.Dropout(0.2)(x)
    x = Dense(1, kernel_regularizer=l1_l2(l1=0.001, l2=0.001), activation='linear')(x)
    inputs = img_input

	# Create model
    model = Model(inputs, x, name='xception')

	# Download and cache the Xception weights file
	#weights_path = get_file('xception_weights.h5', WEIGHTS_PATH, cache_subdir='models')

	# load weights
	#model.load_weights(weights_path)

    return model



model =Xception()
epochs = 25
drop_rate = 0.1
lr = 0.001
#model = densenet_3d(1, input_shape, dropout_rate=drop_rate)
#model = resnet(input_shape)
#model = CNNModel()
opt = Adam(lr=0.001, decay=0.1)
opt1 = tf.keras.optimizers.Adamax(
    learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, name="Adamax")
#opt = tf.keras.optimizers.RMSprop(learning_rate=0.001, momentum=0.1)
#opt = Ranger(learning_rate=0.001, weight_decay=0.01)

#1RAdam(learning_rate=0.001, decay = 0.5)
#2RAdam(learning_rate=0.0005, decay = 0.5)
#3RAdam(learning_rate=0.0001, decay = 0.001)
#4opt = RAdam(learning_rate=0.001, decay = 0.001)
opt = RAdam(learning_rate=0.0001, decay=0.05)
#lr = 0.001 - decay = 0.01 (apprentissage rapide - validation non stable)
#lr = 0.001 - decay = 0.01

rmse = tf.keras.metrics.RootMeanSquaredError()
#run_opts = tf.compat.v1.RunOptions(report_tensor_allocations_upon_oom = True)
print("finished2")

sgd = SGD(lr=lr, momentum=0.90, nesterov=True)
loss = tf.keras.losses.Huber()
loss = tf.keras.losses.mean_squared_error

model.compile(loss=loss, optimizer=opt, metrics=['mae', rmse, 'mse'])


model.summary()

print('Start training..')

from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.callbacks import EarlyStopping
from keras.callbacks import CSVLogger
import pandas as pd

# every epoch check validation accuracy scores and save the highest
checkpoint_2 = ModelCheckpoint('weights-{epoch:02d}.h5',
                               monitor='val_root_mean_squared_error',
                               verbose=1, save_best_only=False)
# every 10 epochs save weights
checkpoint = ModelCheckpoint('weights_SE-XCEPTION2_{epoch:02d}.h5',
                             monitor='val_root_mean_squared_error',
                             verbose=1, save_best_only=False)
history_checkpoint = CSVLogger("SE-xception2.csv", append=True)

# use tensorboard can watch the change in time
tensorboard_ = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)

early_stopping = EarlyStopping(monitor='val_root_mean_squared_error', patience=5, verbose=1, mode='auto')
CONTINUE_TRAINING = False
if (CONTINUE_TRAINING == True):
    history = pd.read_csv('SE-xception2.csv')
    history = history.tail(4)
    INITIAL_EPOCH = history.shape[0]
    #model.load_weights('weights_XCEPTION1_{epoch:02d}.h5' % INITIAL_EPOCH)
    model.load_weights('weights_SE-XCEPTION1_%02d.h5' % INITIAL_EPOCH)
    checkpoint.best = np.min(history['val_root_mean_squared_error'])
else:
    INITIAL_EPOCH = 0

history = model.fit(train_data, epochs=100,
                                  steps_per_epoch= len(train_data.filenames) // 50,
                                  validation_data=test_data, validation_steps=len(test_data.filenames) // 450,
                                callbacks=[history_checkpoint])

values = history.history
validation_loss = values['val_loss']
validation_mae = values['val_mae']
training_mae = values['mae']
validation_rmse = values['val_root_mean_squared_error']
training_rmse = values['root_mean_squared_error']
training_loss = values['loss']

epochs = range(2000)

plt.plot(epochs, training_loss, label='Training Loss')
plt.plot(epochs, validation_loss, label='Validation Loss')
plt.title('Epochs vs Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(epochs, training_mae, label='Training MAE')
plt.plot(epochs, validation_mae, label='Validation MAE')
plt.title('Epochs vs MAE')
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.legend()
plt.show()

plt.plot(epochs, training_rmse, label='Training RMSE')
plt.plot(epochs, validation_rmse, label='Validation RMSE')
plt.title('Epochs vs RMSE')
plt.xlabel('Epochs')
plt.ylabel('RMSE')
plt.legend()
plt.show()
plt.show()


# scores = model.evaluate(train_data)
#scores = model.predict_generator(test_data, len(test_data.filenames) // 750)
#print("%s: %.2f" % (model.metrics_names[1], scores[1]), "%s: %.2f" % (model.metrics_names[2], scores[2]))