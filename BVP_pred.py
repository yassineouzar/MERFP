import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, BatchNormalization, Activation, Flatten, Dropout, AveragePooling3D, Conv3D, add, GlobalAveragePooling3D, multiply, Reshape
from tensorflow.keras.metrics import  RootMeanSquaredError
from keras_radam import RAdam
from tensorflow.keras.regularizers import l1_l2
from Generator_overlap_BVP_norm import ImageDataGenerator
from DepthwiseConv3D1 import DepthwiseConv3D
SeparableConv3D = DepthwiseConv3D

from inference_preprocess import detrend
from scipy.signal import butter, filtfilt
import numpy as np

batch_size = 1
datagen = ImageDataGenerator()

prediction_data = datagen.flow_from_directory(directory='/home/ouzar1/Documents/pythonProject/MMSE/ROI',
                                              label_dir='/home/ouzar1/Documents/pythonProject/MMSE/HR',
                                              target_size=(160, 120), class_mode='label', batch_size=batch_size,
                                              frames_per_step=100, shuffle=False)


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

def vgg():
    # Determine proper input shape
    # input_shape = _obtain_input_shape(None, default_size=299, min_size=71, data_format='channels_last', include_top=False)

    img_input = Input(shape=(100, 160, 120, 3))

    # Block 1
    x = Conv3D(32, (3, 3, 3), strides=(1, 2, 2), padding='same', kernel_regularizer=l1_l2(l1=0.01, l2=0.01),
               use_bias=False)(img_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = AveragePooling3D((3, 3, 3), strides=(1, 2, 2), padding='same')(x)

    x = Conv3D(64, (3, 3, 3), strides=(1, 2, 2), padding='same', kernel_regularizer=l1_l2(l1=0.01, l2=0.01),
               use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = AveragePooling3D((3, 3, 3), strides=(1, 2, 2), padding='same')(x)

    x = Conv3D(128, (3, 3, 3), strides=(1, 2, 2), padding='same', kernel_regularizer=l1_l2(l1=0.01, l2=0.01),
               use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = AveragePooling3D((3, 3, 3), strides=(1, 2, 2), padding='same')(x)

    x = Conv3D(1, (3, 3, 3), strides=(1, 2, 2), padding='same', kernel_regularizer=l1_l2(l1=0.01, l2=0.01),
               use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = AveragePooling3D((3, 3, 3), strides=(1, 2, 2), padding='same')(x)

    # Fully Connected Layer
    #x = GlobalAveragePooling3D()(x)

    x = Flatten()(x)
    # x = keras.layers.Dense(1024,kernel_regularizer=l1_l2(l1=0.01, l2=0.01), activation='relu')(x)
    # x = keras.layers.Dropout(0.4)(x)
    # x = Dense(1, kernel_regularizer=l1_l2(l1=0.01, l2=0.01), activation='linear')(x)
    # x = Dense(128, activation='relu')(x)
    # x = Dropout(0.2)(x)
    x = Dense(100, activation='linear')(x)##sortie à changer
    inputs = img_input

    # Create model
    model = Model(inputs, x, name='xception')

    # Download and cache the Xception weights file
    # weights_path = get_file('xception_weights.h5', WEIGHTS_PATH, cache_subdir='models')

    # load weights
    # model.load_weights(weights_path)
    return model


def Xception():

	# Determine proper input shape
	#input_shape = _obtain_input_shape(None, default_size=299, min_size=71, data_format='channels_last', include_top=False)


    img_input = Input(shape=(100, 160, 120, 3))

	# Block 1
    x = Conv3D(8, (3, 3, 3), strides=(2, 2, 2),kernel_regularizer=l1_l2(l1=0.01, l2=0.01), use_bias=False)(img_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv3D(16, (3, 3, 3),kernel_regularizer=l1_l2(l1=0.01, l2=0.01),use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    residual = Conv3D(32, (1, 1, 1), strides=(2, 2, 2), padding='same',kernel_regularizer=l1_l2(l1=0.01, l2=0.01),use_bias=False)(x)
    residual = BatchNormalization()(residual)

    # Block 2
    x = SeparableConv3D(kernel_size=(3, 3, 3), depth_multiplier=1)(x)

    #x = SeparableConv3D(128, (3, 3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    #x = SeparableConv3D(128, (3, 3, 3), padding='same', use_bias=False)(x)
    x = SeparableConv3D(kernel_size = (3, 3, 3), depth_multiplier = 2)(x)
    x = BatchNormalization()(x)

    # Block 2 Pool
    x = AveragePooling3D((3, 3, 3), strides=(2, 2, 2), padding='same')(x)
    x = add([x, residual])

    residual = Conv3D(64, (1, 1, 1), strides=(2, 2, 2), padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

	# Block 3
    x = Activation('relu')(x)
    x = SeparableConv3D(kernel_size = (3, 3, 3), depth_multiplier = 2)(x)

    #x = SeparableConv3D(256, (3, 3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv3D(kernel_size = (3, 3, 3), depth_multiplier = 1)(x)

    #x = SeparableConv3D(256, (3, 3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)

    # Block 3 Pool
    x = AveragePooling3D((3, 3, 3), strides=(2, 2, 2), padding='same')(x)

    x = add([x, residual])

    residual = Conv3D(256, (1, 1, 1), strides=(2, 2, 2), padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    # Block 4
    x = Activation('relu')(x)
    x = SeparableConv3D(kernel_size = (3, 3, 3), depth_multiplier = 2)(x)

    #x = SeparableConv3D(728, (3, 3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv3D(kernel_size = (3, 3, 3), depth_multiplier = 2)(x)

    #x = SeparableConv3D(768, (3, 3, 3), padding='same', use_bias=False)(x)

    x = BatchNormalization()(x)

    x = AveragePooling3D((3, 3, 3), strides=(2, 2, 2), padding='same')(x)

    x = add([x, residual])

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
        x = add([x, residual])

    residual = Conv3D(256, (1, 1, 1), strides=(2, 2, 2), padding='same',kernel_regularizer=l1_l2(l1=0.01, l2=0.01),use_bias=False)(x)
    residual = BatchNormalization()(residual)
	# Block 13
    x = Activation('relu')(x)
    x = SeparableConv3D(kernel_size=(3, 3, 3), depth_multiplier=1)(x)

    #x = SeparableConv3D(728, (3, 3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv3D(kernel_size=(3, 3, 3), depth_multiplier=1)(x)

    #x = SeparableConv3D(1024, (3, 3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)

	# Block 13 Pool
    x = AveragePooling3D((3, 3, 3), strides=(2, 2, 2), padding='same')(x)

    x = add([x, residual])

	# Block 14
    x = SeparableConv3D(kernel_size=(3, 3, 3), depth_multiplier=1)(x)

    #x = SeparableConv3D(1536, (3, 3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

	# Block 14 part 2
    x = SeparableConv3D(kernel_size=(3, 3, 3), depth_multiplier=2)(x)

    #x = SeparableConv3D(2048, (3, 3, 3), padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = AveragePooling3D((3, 3, 3), strides=(2, 2, 2), padding='same')(x)

    # Fully Connected Layer
    x = GlobalAveragePooling3D()(x)

    #x = Flatten()(x)
    #x = keras.layers.Dense(1024,kernel_regularizer=l1_l2(l1=0.01, l2=0.01), activation='relu')(x)
    #x = keras.layers.Dropout(0.4)(x)
    #x = Dense(1, kernel_regularizer=l1_l2(l1=0.01, l2=0.01), activation='linear')(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(100, activation='linear')(x)##sortie à changer
    inputs = img_input

	# Create model
    model = Model(inputs, x, name='xception')

	# Download and cache the Xception weights file
	#weights_path = get_file('xception_weights.h5', WEIGHTS_PATH, cache_subdir='models')

	# load weights
	#model.load_weights(weights_path)

    return model

def SE_Xception():

	# Determine proper input shape
	#input_shape = _obtain_input_shape(None, default_size=299, min_size=71, data_format='channels_last', include_top=False)


    img_input = Input(shape=(prediction_data.frames_per_step, 160, 120, 3))

    # Block 1
    d1 = Conv3D(32, (3, 3, 3), activation='tanh', strides=(1, 2, 2), padding='same',
                kernel_regularizer=l1_l2(l1=0.001, l2=0.001),
                use_bias=True)(
        img_input)
    d1 = BatchNormalization()(d1)
    # d1 = Activation('relu')(d1)
    d2 = Conv3D(32, (3, 3, 3), activation='tanh', strides=(1, 2, 2), padding='same',
                kernel_regularizer=l1_l2(l1=0.001, l2=0.001),
                use_bias=True)(
        d1)
    d2 = BatchNormalization()(d2)

    d3 = AveragePooling3D((3, 3, 3), strides=(1, 2, 2), padding='same')(d2)
    d4 = Dropout(0.2)(d3)

    d5 = Conv3D(64, (3, 3, 3), activation='tanh', strides=(1, 2, 2), padding='same', kernel_regularizer=l1_l2(l1=0.001, l2=0.001),
                use_bias=True)(d4)
    d5 = BatchNormalization()(d5)
    # d5 = Activation('relu')(d5)
    d6 = Conv3D(64, (3, 3, 3), activation='tanh', strides=(1, 2, 2), padding='same', kernel_regularizer=l1_l2(l1=0.001, l2=0.001),
                use_bias=True)(d5)
    d6 = BatchNormalization()(d6)

    d7 = AveragePooling3D((3, 3, 3), strides=(1, 2, 2), padding='same')(d6)
    d8 = Dropout(0.2)(d7)

    d9 = Conv3D(128, (3, 3, 3), activation='tanh', strides=(1, 2, 2), padding='same', kernel_regularizer=l1_l2(l1=0.001, l2=0.001),
                use_bias=True)(d8)
    d9 = BatchNormalization()(d9)
    # d5 = Activation('relu')(d5)
    d10 = Conv3D(128, (3, 3, 3), activation='tanh', strides=(1, 2, 2), padding='same', kernel_regularizer=l1_l2(l1=0.001, l2=0.001),
                use_bias=True)(d9)
    d10 = BatchNormalization()(d10)

    se = se_block(d10, num_filters=128)

    d11 = AveragePooling3D((3, 3, 3), strides=(1, 2, 2), padding='same')(se)
    d12 = Dropout(0.2)(d11)

    # Fully Connected Layer
    # x = GlobalAveragePooling3D()(x)

    x = Flatten()(d12)
    #x = keras.layers.Dense(1024,kernel_regularizer=l1_l2(l1=0.01, l2=0.01), activation='relu')(x)
    #x = keras.layers.Dropout(0.4)(x)
    #x = Dense(1, kernel_regularizer=l1_l2(l1=0.01, l2=0.01), activation='linear')(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(prediction_data.frames_per_step, activation='linear')(x)##sortie à changer
    inputs = img_input

	# Create model
    model = Model(inputs, x, name='xception')

	# Download and cache the Xception weights file
	#weights_path = get_file('xception_weights.h5', WEIGHTS_PATH, cache_subdir='models')

	# load weights
	#model.load_weights(weights_path)

    return model


model = SE_Xception()

# model = vgg()
# model = Xception()

model.summary()


opt = RAdam(learning_rate=0.0001, decay=0.01)
rmse = RootMeanSquaredError()

model.compile(loss='mse', optimizer=opt, metrics=['mae', rmse])
# model.load_weights("/home/ouzar1/Documents/pythonProject/weights.15-0.2665.h5")
model.load_weights("/home/ouzar1/Documents/pythonProject/weights.50-0.2653.h5")

#Call Generator

# batch_size = 10
# datagen = ImageDataGenerator()
#
# prediction_data = datagen.flow_from_directory(directory='/home/ouzar1/Documents/pythonProject/MMSE/ROI', label_dir='/home/ouzar1/Documents/pythonProject/MMSE/HR',
#                                          target_size=(160, 120), class_mode='label', batch_size=19,
#                                          frames_per_step=100, shuffle=False)

#Evaluation
# pred = model.evaluate_generator(prediction_data, len(prediction_data.filenames) // 500)
# print("%s: %.2f" % (model.metrics_names[1], pred[1]), "%s: %.2f" % (model.metrics_names[2], pred[2]))


if __name__ == '__main__':

    for data in prediction_data:
        vid = data[0]
        heart_rate = data[1]

        pred = model.predict(vid)
        pulse_pred = detrend(np.cumsum(pred), 100)
        [b_pulse, a_pulse] = butter(1, [0.75/25*2, 2.5/25*2], btype='bandpass')
        pulse_pred = filtfilt(b_pulse, a_pulse, np.double(pulse_pred))
        plt.plot(heart_rate[0])
        plt.plot(pred[0])
        plt.plot(pulse_pred)
        plt.show()
        # print("Pred =  %.2f bpm ", pred, "GT =  %.2f bpm ", heart_rate, "diff =  %.2f bpm ", np.abs(scores-label))



