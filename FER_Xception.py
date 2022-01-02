import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import tensorflow as tf
from tensorflow.python.keras import backend as K

# Disable eager execution (Adam optimizer cannot be used if this option is enabled)
tf.compat.v1.disable_eager_execution()
tf.executing_eagerly()


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, BatchNormalization, Activation, Flatten, Dropout, AveragePooling3D, Conv3D, add, MaxPooling3D, concatenate, GlobalAveragePooling3D, multiply
from tensorflow.keras.metrics import  RootMeanSquaredError
from keras_radam import RAdam
from tensorflow.keras.regularizers import l1_l2, l2
from Generator3 import ImageDataGenerator
from DepthwiseConv3D1 import DepthwiseConv3D
SeparableConv3D = DepthwiseConv3D

import matplotlib.pyplot as plt
import optuna
from Generator_FER import ImageDataGenerator
datagen = ImageDataGenerator()

def Xception():
    # Determine proper input shape
    vid_input = Input(shape=(100, 160, 120, 3))

    # Block 1
    x = Conv3D(8, (3, 3, 3), padding='same', kernel_regularizer=l1_l2(l1=0.01, l2=0.01), use_bias=False)(vid_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv3D(16, (3, 3, 3), padding='same', kernel_regularizer=l1_l2(l1=0.01, l2=0.01), use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    residual = Conv3D(32, (1, 1, 1), padding='same', kernel_regularizer=l1_l2(l1=0.01, l2=0.01),
                      use_bias=False)(x)
    residual = BatchNormalization()(residual)

    # Block 2
    # x = SeparableConv3D(kernel_size=(3, 3, 3), depth_multiplier=1)(x)
    # x = BatchNormalization()(x)
    # x = Activation('relu')(x)
    x = SeparableConv3D(kernel_size=(3, 3, 3), depth_multiplier=2)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = add([x, residual])

    # Block 2 Pool
    x = MaxPooling3D((3, 3, 3), strides=(1, 5, 5), padding='same')(x)

    residual = Conv3D(64, (1, 1, 1), padding='same', kernel_regularizer=l1_l2(l1=0.01, l2=0.01),
                      use_bias=False)(x)
    residual = BatchNormalization()(residual)

    # Block 3
    x = SeparableConv3D(kernel_size=(3, 3, 3), depth_multiplier=2)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # x = SeparableConv3D(kernel_size=(3, 3, 3), depth_multiplier=1)(x)
    # x = BatchNormalization()(x)
    # x = Activation('relu')(x)

    x = add([x, residual])

    # Block 3 Pool
    x = MaxPooling3D((3, 3, 3), strides=(1, 2, 2), padding='same')(x)

    residual = Conv3D(128, (1, 1, 1), padding='same', kernel_regularizer=l1_l2(l1=0.01, l2=0.01),
                      use_bias=False)(x)
    residual = BatchNormalization()(residual)

    # Block 4
    x = SeparableConv3D(kernel_size=(3, 3, 3), depth_multiplier=2)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # x = SeparableConv3D(kernel_size=(3, 3, 3), depth_multiplier=1)(x)
    # x = BatchNormalization()(x)
    # x = Activation('relu')(x)

    x = add([x, residual])

    # Block 4 Pool
    x = MaxPooling3D((3, 3, 3), strides=(1, 2, 2), padding='same')(x)

    # Block 5 - 12
    for i in range(8):
        residual = x

        x = SeparableConv3D(kernel_size=(3, 3, 3), depth_multiplier=1)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        # x = SeparableConv3D(kernel_size=(3, 3, 3), depth_multiplier=1)(x)
        # x = BatchNormalization()(x)
        # x = Activation('relu')(x)
        # x = SeparableConv3D(kernel_size=(3, 3, 3), depth_multiplier=1)(x)
        # x = BatchNormalization()(x)
        # x = Activation('relu')(x)

        x = add([x, residual])

    residual = Conv3D(256, (1, 1, 1), padding='same', kernel_regularizer=l1_l2(l1=0.01, l2=0.01),
                      use_bias=False)(x)
    residual = BatchNormalization()(residual)

    # Block 13
    x = SeparableConv3D(kernel_size=(3, 3, 3), depth_multiplier=2)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    # x = SeparableConv3D(kernel_size=(3, 3, 3), depth_multiplier=1)(x)
    # x = BatchNormalization()(x)
    # x = Activation('relu')(x)

    x = add([x, residual])

    # Block 13 Pool
    x = MaxPooling3D((3, 3, 3), strides=(1, 2, 2), padding='same')(x)

    # Block 14
    x = SeparableConv3D(kernel_size=(3, 3, 3), depth_multiplier=1)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Block 14 part 2
    # x = SeparableConv3D(kernel_size=(3, 3, 3), depth_multiplier=2)(x)
    # x = BatchNormalization()(x)
    # x = Activation('relu')(x)
    x = MaxPooling3D((3, 3, 3), strides=(1, 5, 5), padding='same')(x)

    # residual = Conv3D(512, (1, 1, 1), padding='same', kernel_regularizer=l1_l2(l1=0.01, l2=0.01),
    #                   use_bias=False)(x)
    # x = BatchNormalization()(x)
    # x = Activation('relu')(x)
    # x = MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same')(x)

    # Fully Connected Layer
    x = Flatten()(x)
    x = Dense(256, kernel_regularizer=l1_l2(l1=0.01, l2=0.01), activation='relu')(x)
    # x = Dropout(0.2)(x)
    x = Dense(4, kernel_regularizer=l1_l2(l1=0.01, l2=0.01), activation='softmax')(x)

    inputs = vid_input

    # Create model
    model = Model(inputs, x, name='xception')

    return model


nb_classes = 4
input_shape = (100, 160, 120, 3)

model = Xception()
model.summary()
epochs = 25
drop_rate = 0.1
lr = 0.01
#model = densenet_3d(1, input_shape, dropout_rate=drop_rate)
#model = resnet(input_shape)
#model = CNNModel()


opt = RAdam(learning_rate=0.0001, decay=0.01)
rmse = RootMeanSquaredError()

model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])


print('Start training..')

from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import CSVLogger
import pandas as pd

# every epoch check validation accuracy scores and save the highest
checkpoint_2 = ModelCheckpoint('weights-{epoch:02d}.h5',
                               monitor='val_root_mean_squared_error',
                               verbose=1, save_best_only=False)
# every 10 epochs save weights
checkpoint = ModelCheckpoint('weights.{epoch:02d}-{val_accuracy:.4f}.h5',
                             monitor='val_accuracy',
                             verbose=1, save_best_only=True)
history_checkpoint = CSVLogger("hist_F1_50ep.csv", append=True)

# use tensorboard can watch the change in time
tensorboard_ = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)

early_stopping = EarlyStopping(monitor='val_root_mean_squared_error', patience=5, verbose=1, mode='auto')
"""
if (CONTINUE_TRAINING == True):
    history = pd.read_csv('history2.csv')
    INITIAL_EPOCH = history.shape[0]
    model.load_weights('weights_%02d.h5' % INITIAL_EPOCH)
    checkpoint_2.best = np.min(history['val_root_mean_squared_error'])
else:
    INITIAL_EPOCH = 0
"""

batch_size=1
#datagen = ImageDataGenerator()
#train_data = datagen.flow_from_directory('D:/HR_estimation/ROI1', 'D:/HR_estimation/heart_rate', target_size=(120, 120), class_mode='label', batch_size=1, frames_per_step=75, shuffle=False)
#test_data=datagen.flow_from_directory('D:/HR_estimation/ROI2', 'D:/HR_estimation/heart_rate2', target_size=(120, 120), class_mode='label', batch_size=1, frames_per_step=75)
train_data = datagen.flow_from_directory(directory='/home/ouzar1/Documents/pythonProject/BP4D-Expressive-Segment/multimodal/FE/Train_data',
                                             target_size=(160, 120), class_mode='categorical', batch_size=4,
                                             frames_per_step=100, shuffle=False)
test_data = datagen.flow_from_directory(directory='/home/ouzar1/Documents/pythonProject/BP4D-Expressive-Segment/BP4D-Expressive-Segment/Test_set',
                                             target_size=(160, 120), class_mode='categorical', batch_size=1,
                                             frames_per_step=100, shuffle=False)

history = model.fit(train_data, epochs=50,
                                  steps_per_epoch= len(train_data.filenames) // 400,
                                  validation_data=test_data, validation_steps=len(test_data.filenames) // 100, callbacks=[history_checkpoint, checkpoint])
#, callbacks=[history_checkpoint, checkpoint]
values = history.history
validation_loss = values['val_loss']
validation_accuracy = values['val_accuracy']
training_accuracy = values['accuracy']
training_loss = values['loss']

epochs = range(50)

plt.plot(epochs, training_loss, label='Training Loss')
plt.plot(epochs, validation_loss, label='Validation Loss')
plt.title('Epochs vs Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(epochs, training_accuracy, label='Training accuracy')
plt.plot(epochs, validation_accuracy, label='Validation accuracy')
plt.title('Epochs vs accuracy')
plt.xlabel('Epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()

scores = model.evaluate(test_data, len(train_data.filenames) // 200)
#scores = model.predict_generator(train_data, len(train_data.filenames) // 200)
#print("%s: %f" % (model.metrics_names[1], scores[1]))
print(scores)
# scores = model.evaluate(train_data) ,kernel_regularizer=l2(0.001)



