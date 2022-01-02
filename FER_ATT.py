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


class Attention_mask(tf.keras.layers.Layer):
    def call(self, x):
        xsum = K.sum(x, axis=1, keepdims=True)
        xsum = K.sum(xsum, axis=2, keepdims=True)
        xshape = K.int_shape(x)
        return x / xsum * xshape[1] * xshape[2] * 0.5

    def get_config(self):
        config = super(Attention_mask, self).get_config()
        return config


def conv_factory(x, nb_filter, kernel=(3,3,3), weight_decay=0.01):
    x = BatchNormalization(axis=-1, epsilon=1.1e-5)(x)
    x = Activation('relu')(x)
    x = Conv3D(nb_filter, kernel,
               kernel_initializer='he_normal',
               padding="same",
               use_bias=False,
               kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(x)
    return x


def residual_block(x, filters, drop_rate=0., weight_decay=0.01):

    x = conv_factory(x, 4 * filters, kernel=(1, 1, 1))
    if drop_rate:
        x = Dropout(drop_rate)(x)
    x = conv_factory(x, filters, kernel=(3, 3, 3))
    if drop_rate:
        x = Dropout(drop_rate)(x)
    x = Conv3D(4 * filters, (1, 1, 1),
               kernel_initializer='he_normal',
               padding="same",
               use_bias=False,
               kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(x)
    return x


def resnet_3d(nb_classes, input_shape, drop_rate=0., weight_decay=0.005):

    model_input = Input(shape=input_shape)

    # Appearance Branch
    r1 = Conv3D(nb_filters1, kernel_size, padding='same', activation='tanh')(rawf_input)
    r2 = Conv3D(nb_filters1, kernel_size, activation='tanh')(r1)
    g1 = Conv3D(1, (1, 1, 1), padding='same', activation='sigmoid')(r2)
    g1 = Attention_mask()(g1)
    gated1 = multiply([d2, g1])

    # 112x112x8
    # stage 1 Initial convolution
    x = Conv3D(64, (3, 3, 3),
               kernel_initializer="he_normal",
               padding="same",activation='tanh',
               use_bias=False,
               kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(model_input)
    x = BatchNormalization(axis=-1, epsilon=1.1e-5)(x)
    x = Activation('relu')(x)
    x = MaxPooling3D((2, 2, 2), strides=(1, 4, 3), padding='same')(x)
    # 56x56x8

    # stage 2 convolution
    y = Conv3D(128, (1, 1, 1),
               kernel_initializer="he_normal",
               padding="same",
               use_bias=False,
               kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(x)

    x = residual_block(x, 32, drop_rate=drop_rate)
    y = add([x, y])
    y = MaxPooling3D((2, 2, 2), strides=(1, 2, 2), padding='same')(y)
    # 28x28x4

    # stage 3
    x = residual_block(y, 32, drop_rate=drop_rate)
    y = add([x, y])
    x = MaxPooling3D((2, 2, 2), strides=(1, 5, 5), padding='same')(y)
    # 14x14x2

    # stage 4
    y = BatchNormalization(axis=-1, epsilon=1.1e-5)(x)
    y = Activation('relu')(y)
    y = Conv3D(256, (1, 1, 1),
                kernel_initializer='he_normal',
                padding="same",
                use_bias=False,
                kernel_regularizer=l1_l2(l1=0.01, l2=0.01))(y)

    x = residual_block(x, 64, drop_rate=drop_rate)
    y = add([x, y])
    x = residual_block(y, 64, drop_rate=drop_rate)
    y = add([x, y])
    y = MaxPooling3D((2, 2, 2), strides=(1, 4, 4), padding='same')(y)
    # 7x7x1

    # stage 5

    x = residual_block(y, 64, drop_rate=drop_rate)
    y = add([x, y])
    x = residual_block(y, 64, drop_rate=drop_rate)
    y = add([x, y])

    x = BatchNormalization(axis=-1, epsilon=1.1e-5)(y)
    x = Activation('relu')(x)

    # x = GlobalAveragePooling3D()(x)

    x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.2)(x)

    x = Dense(nb_classes,
              activation='softmax',
              kernel_regularizer=l1_l2(l1=0.01, l2=0.01),
              bias_regularizer=l1_l2(l1=0.01, l2=0.01))(x)

    model = Model(inputs=model_input, outputs=x, name="resnet_3d")

    return model

def MT_CAN_3D(nb_classes = 4, kernel_size=(3, 3, 3), dropout_rate1=0.25,
              dropout_rate2=0.5, pool_size=(2, 2, 2), nb_dense=128):
    # Determine proper input shape


    model_input = Input(shape=(100, 160, 120, 3))

    # Appearance Branch
    r1 = Conv3D(32, kernel_size, padding='same', activation='tanh')(model_input)
    r2 = Conv3D(32, kernel_size, padding='same', activation='tanh')(r1)
    g1 = Conv3D(1, (1, 1, 1), padding='same', activation='sigmoid')(r2)
    g1 = Attention_mask()(g1)

    d3 = Dropout(dropout_rate1)(g1)
    d4 = Conv3D(32, kernel_size, padding='same', activation='tanh')(d3)
    d5 = Conv3D(32, kernel_size, padding='same', activation='tanh')(d4)

    gated1 = multiply([d5, g1])

    d6 = MaxPooling3D(pool_size=(1, 4, 3))(gated1)
    d7 = Dropout(dropout_rate1)(d6)

    d8 = Conv3D(64, kernel_size, padding='same', activation='tanh')(d7)
    d9 = Conv3D(64, kernel_size, padding='same', activation='tanh')(d8)
    g2 = Conv3D(1, (1, 1, 1), padding='same', activation='sigmoid')(d9)
    g2 = Attention_mask()(g2)

    d10 = Dropout(dropout_rate1)(g2)
    d11 = Conv3D(64, kernel_size, padding='same', activation='tanh')(d10)
    d12 = Conv3D(64, kernel_size, padding='same', activation='tanh')(d11)

    gated2 = multiply([d12, g2])

    d13 = MaxPooling3D(pool_size=(1, 2, 2))(gated2)
    d14 = Dropout(dropout_rate1)(d13)

    d15 = Conv3D(128, kernel_size, padding='same', activation='tanh')(d14)
    d16 = Conv3D(128, kernel_size, padding='same', activation='tanh')(d15)
    g3 = Conv3D(1, (1, 1, 1), padding='same', activation='sigmoid')(d16)
    g3 = Attention_mask()(g3)

    d17 = Dropout(dropout_rate1)(g3)
    d18 = Conv3D(128, kernel_size, padding='same', activation='tanh')(d17)
    d19 = Conv3D(128, kernel_size, padding='same', activation='tanh')(d18)

    gated3 = multiply([d19, g3])

    d20 = MaxPooling3D(pool_size=(1, 4, 4))(gated3)
    d21 = Dropout(dropout_rate1)(d20)

    d22 = Conv3D(256, kernel_size, padding='same', activation='tanh')(d21)
    d23 = Conv3D(256, kernel_size, padding='same', activation='tanh')(d22)
    g4 = Conv3D(1, (1, 1, 1), padding='same', activation='sigmoid')(d23)
    g4 = Attention_mask()(g4)

    d24= Dropout(dropout_rate1)(g4)
    d25 = Conv3D(256, kernel_size, padding='same', activation='tanh')(d24)
    d26 = Conv3D(256, kernel_size, padding='same', activation='tanh')(d25)

    gated4 = multiply([d26, g4])

    d27 = MaxPooling3D(pool_size=(1, 5, 5))(gated4)
    d28 = Dropout(dropout_rate1)(d27)

    d29 = Flatten()(d28)
    d10_y = Dense(1024, activation='tanh')(d29)
    d11_y = Dropout(dropout_rate2)(d10_y)
    out_y = Dense(nb_classes,
              activation='softmax',
              kernel_regularizer=l1_l2(l1=0.01, l2=0.01),
              bias_regularizer=l1_l2(l1=0.01, l2=0.01))(d11_y)

    model = Model(inputs=model_input, outputs=out_y, name="Attention")

    return model

nb_classes = 4
input_shape = (100, 160, 120, 3)

model = MT_CAN_3D()
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



