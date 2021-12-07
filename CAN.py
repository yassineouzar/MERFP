import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import tensorflow as tf
from tensorflow.python.keras import backend as K

# Disable eager execution (Adam optimizer cannot be used if this option is enabled)
tf.compat.v1.disable_eager_execution()
tf.executing_eagerly()

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, BatchNormalization, Activation, Flatten, Dropout, AveragePooling3D, Conv3D, add, concatenate, Conv2D, AveragePooling2D, multiply, Reshape, GlobalAveragePooling3D
from tensorflow.keras.metrics import  RootMeanSquaredError
from keras_radam import RAdam
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.optimizers import Adam
from DepthwiseConv3D1 import DepthwiseConv3D
SeparableConv3D = DepthwiseConv3D

import matplotlib.pyplot as plt


from Generator_diff import ImageDataGenerator
datagen = ImageDataGenerator()

train = datagen.flow_from_directory(
    directory='/home/ouzar1/Documents/pythonProject/BP4D-Expressive-Segment/multimodal/FE/Train_data',
    label_dir='/home/ouzar1/Documents/pythonProject/BP4D-Expressive-Segment/multimodal/PPG/Train_data',
    target_size=(160, 120), class_mode='label', batch_size=1,
    frames_per_step=50, shuffle=False)

test = datagen.flow_from_directory(
    directory='/home/ouzar1/Documents/pythonProject/BP4D-Expressive-Segment/multimodal/FE/Test_data',
    label_dir='/home/ouzar1/Documents/pythonProject/BP4D-Expressive-Segment/multimodal/PPG/Test_data',
    target_size=(160, 120), class_mode='label', batch_size=1,
    frames_per_step=50, shuffle=False)



from Generator_BVP import ImageDataGenerator
datagen1 = ImageDataGenerator()
def train_generator_multiple():
    genX1 = datagen1.flow_from_directory(directory='/home/ouzar1/Documents/pythonProject/BP4D-Expressive-Segment/multimodal/FE/Train_data',
                                              label_dir='/home/ouzar1/Documents/pythonProject/BP4D-Expressive-Segment/multimodal/PPG/Train_data',
                                             target_size=(160, 120), class_mode='label', batch_size=1,
                                             frames_per_step=50, shuffle=False)

    genX2 = datagen1.flow_from_directory(directory='/home/ouzar1/Documents/pythonProject/BP4D-Expressive-Segment/multimodal/FE/Train_data',
                                              label_dir='/home/ouzar1/Documents/pythonProject/BP4D-Expressive-Segment/multimodal/PPG/Train_data',
                                             target_size=(160, 120), class_mode='label', batch_size=1,
                                             frames_per_step=50, shuffle=False)

    while True:
        X1i = genX1.next()
        X2i = genX2.next()
        yield [X1i[0], X2i[0]], X2i[1]  # Yield both images and their mutual label

def test_generator_multiple():
    genX1 = datagen.flow_from_directory(directory='/home/ouzar1/Documents/pythonProject/BP4D-Expressive-Segment/multimodal/FE/Test_data',
                                              label_dir='/home/ouzar1/Documents/pythonProject/BP4D-Expressive-Segment/multimodal/PPG/Test_data',
                                             target_size=(160, 120), class_mode='label', batch_size=1,
                                             frames_per_step=50, shuffle=False)

    genX2 = datagen1.flow_from_directory(directory='/home/ouzar1/Documents/pythonProject/BP4D-Expressive-Segment/multimodal/FE/Test_data',
                                              label_dir='/home/ouzar1/Documents/pythonProject/BP4D-Expressive-Segment/multimodal/PPG/Test_data',
                                             target_size=(160, 120), class_mode='label', batch_size=1,
                                             frames_per_step=50, shuffle=False)

    while True:
        X1i = genX1.next()
        X2i = genX2.next()
        yield [X1i[0], X2i[0]], X2i[1]  # Yield both images and their mutual label

train_generator = train_generator_multiple()
test_generator = test_generator_multiple()

#for data in train_generator:
    #image = data[0]
    #label = data[1]
    #print(image[0].shape, image[1].shape, label.shape)

class TSM(tf.keras.layers.Layer):
    def call(self, x, n_frame, fold_div=3):
        nt, h, w, c = x.shape
        x = K.reshape(x, (-1, n_frame, h, w, c))
        fold = c // fold_div
        last_fold = c - (fold_div - 1) * fold
        out1, out2, out3 = tf.split(x, [fold, fold, last_fold], axis=-1)

        # Shift left
        padding_1 = tf.zeros_like(out1)
        padding_1 = padding_1[:, -1, :, :, :]
        padding_1 = tf.expand_dims(padding_1, 1)
        _, out1 = tf.split(out1, [1, n_frame - 1], axis=1)
        out1 = tf.concat([out1, padding_1], axis=1)

        # Shift right
        padding_2 = tf.zeros_like(out2)
        padding_2 = padding_2[:, 0, :, :, :]
        padding_2 = tf.expand_dims(padding_2, 1)
        out2, _ = tf.split(out2, [n_frame - 1, 1], axis=1)
        out2 = tf.concat([padding_2, out2], axis=1)

        out = tf.concat([out1, out2, out3], axis=-1)
        out = K.reshape(out, (-1, h, w, c))

        return out

    def get_config(self):
        config = super(TSM, self).get_config()
        return config

class Attention_mask(tf.keras.layers.Layer):
    def call(self, x):
        xsum = K.sum(x, axis=1, keepdims=True)
        xsum = K.sum(xsum, axis=2, keepdims=True)
        xshape = K.int_shape(x)
        return x / xsum * xshape[1] * xshape[2] * 0.5

    def get_config(self):
        config = super(Attention_mask, self).get_config()
        return config


def TSM_Cov2D(x, n_frame, nb_filters=128, kernel_size=(3, 3), activation='tanh', padding='same'):
    x = TSM()(x, n_frame)
    x = Conv2D(nb_filters, kernel_size, padding=padding, activation=activation)(x)
    return x

def MTTS_CAN(n_frame, nb_filters1, nb_filters2, input_shape, kernel_size=(3, 3), dropout_rate1=0.25,
             dropout_rate2=0.5, pool_size=(2, 2), nb_dense=128):
    diff_input = Input(shape=input_shape)
    rawf_input = Input(shape=input_shape)

    d1 = TSM_Cov2D(diff_input, n_frame, nb_filters1, kernel_size, padding='same', activation='tanh')
    d2 = TSM_Cov2D(d1, n_frame, nb_filters1, kernel_size, padding='valid', activation='tanh')

    r1 = Conv2D(nb_filters1, kernel_size, padding='same', activation='tanh')(rawf_input)
    r2 = Conv2D(nb_filters1, kernel_size, activation='tanh')(r1)

    g1 = Conv2D(1, (1, 1), padding='same', activation='sigmoid')(r2)
    g1 = Attention_mask()(g1)
    gated1 = multiply([d2, g1])

    d3 = AveragePooling2D(pool_size)(gated1)
    d4 = Dropout(dropout_rate1)(d3)

    r3 = AveragePooling2D(pool_size)(r2)
    r4 = Dropout(dropout_rate1)(r3)

    d5 = TSM_Cov2D(d4, n_frame, nb_filters2, kernel_size, padding='same', activation='tanh')
    d6 = TSM_Cov2D(d5, n_frame, nb_filters2, kernel_size, padding='valid', activation='tanh')

    r5 = Conv2D(nb_filters2, kernel_size, padding='same', activation='tanh')(r4)
    r6 = Conv2D(nb_filters2, kernel_size, activation='tanh')(r5)

    g2 = Conv2D(1, (1, 1), padding='same', activation='sigmoid')(r6)
    g2 = Attention_mask()(g2)
    gated2 = multiply([d6, g2])

    d7 = AveragePooling2D(pool_size)(gated2)
    d8 = Dropout(dropout_rate1)(d7)

    d9 = Flatten()(d8)

    d10_y = Dense(nb_dense, activation='tanh')(d9)
    d11_y = Dropout(dropout_rate2)(d10_y)
    out_y = Dense(1, name='output_1')(d11_y)

    d10_r = Dense(nb_dense, activation='tanh')(d9)
    d11_r = Dropout(dropout_rate2)(d10_r)
    out_r = Dense(50, name='output_2')(d11_r)

    model = Model(inputs=[diff_input, rawf_input], outputs=out_r)

    return model

    y = Conv3D(32, (3, 3, 3), strides=(2, 2, 2), padding='same', kernel_regularizer=l1_l2(l1=0.001, l2=0.001),
               use_bias=True)(
        diff_input)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = Conv3D(32, (3, 3, 3), strides=(2, 2, 2), padding='same', kernel_regularizer=l1_l2(l1=0.001, l2=0.001),
               use_bias=True)(
        y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    att1 = x

    att1 = Conv3D(1, (1, 1, 1), strides=(2, 2, 2), padding='same', kernel_regularizer=l1_l2(l1=0.001, l2=0.001),
                  use_bias=False)(att1)

    att1 = BatchNormalization()(att1)

    # att1 = se_block(att1, num_filters=1)
    att1 = Attention_mask()(att1)

    y1 = multiply([y, att1])
    y1 = AveragePooling3D((3, 3, 3), strides=(2, 2, 2), padding='same')(y1)
    y1 = Dropout(0.2)(y1)

    y1 = Conv3D(64, (3, 3, 3), strides=(2, 2, 2), padding='same', kernel_regularizer=l1_l2(l1=0.001, l2=0.001),
                use_bias=True)(
        y1)
    y1 = BatchNormalization()(y)
    y1 = Activation('relu')(y)
    y1 = Conv3D(64, (3, 3, 3), strides=(2, 2, 2), padding='same', kernel_regularizer=l1_l2(l1=0.001, l2=0.001),
                use_bias=True)(
        y1)
    y1 = BatchNormalization()(y1)
    y1 = Activation('relu')(y1)
    # sum = add([x, att1])
    # relu2 = Activation('relu')(sum)

    x = AveragePooling3D((3, 3, 3), strides=(2, 2, 2), padding='same')(x)
    x = Dropout(0.2)(x)

    x = Conv3D(64, (3, 3, 3), strides=(2, 2, 2), padding='same', kernel_regularizer=l1_l2(l1=0.001, l2=0.001),
               use_bias=True)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv3D(64, (3, 3, 3), strides=(2, 2, 2), padding='same', kernel_regularizer=l1_l2(l1=0.001, l2=0.001),
               use_bias=True)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    att2 = x

    att2 = Conv3D(1, (1, 1, 1), strides=(2, 2, 2), padding='same', kernel_regularizer=l1_l2(l1=0.001, l2=0.001),
                  use_bias=False)(att2)

    att2 = BatchNormalization()(att2)

    att2 = Attention_mask()(att2)

    y2 = multiply([y1, att2])

    # att2 = se_block(att2, num_filters=1)

    # y2 = multiply([att2, y1])
    # y2 = AveragePooling3D((3, 3, 3), strides=(2, 2, 2), padding='same')(y2)
    # y2 = Dropout(0.2)(y2)

    # sum = add([x, att2])
    # relu2 = Activation('relu')(sum)

    y3 = AveragePooling3D((3, 3, 3), strides=(2, 2, 2), padding='same')(y2)
    y3 = Dropout(0.5)(y3)

    y3 = Dense(128, kernel_regularizer=l1_l2(l1=0.001, l2=0.001), activation='relu')(y3)
    y3 = Dropout(0.5)(y3)
    y3 = Dense(50, kernel_regularizer=l1_l2(l1=0.001, l2=0.001), activation='linear')(y3)


def MT_CAN_3D(n_frame, nb_filters1, nb_filters2, input_shape, kernel_size=(3, 3, 3), dropout_rate1=0.25,
              dropout_rate2=0.5, pool_size=(2, 2, 2), nb_dense=128):
    # Determine proper input shape


    diff_input = Input(shape=(50, 160, 120, 3))
    rawf_input = Input(shape=(50, 160, 120, 3))

    d1 = Conv3D(nb_filters1, kernel_size, padding='same', activation='tanh')(diff_input)
    d2 = Conv3D(nb_filters1, kernel_size, activation='tanh')(d1)

    # Appearance Branch
    r1 = Conv3D(nb_filters1, kernel_size, padding='same', activation='tanh')(rawf_input)
    r2 = Conv3D(nb_filters1, kernel_size, activation='tanh')(r1)
    g1 = Conv3D(1, (1, 1, 1), padding='same', activation='sigmoid')(r2)
    g1 = Attention_mask()(g1)
    gated1 = multiply([d2, g1])

    d3 = AveragePooling3D(pool_size)(gated1)
    d4 = Dropout(dropout_rate1)(d3)
    d5 = Conv3D(nb_filters2, kernel_size, padding='same', activation='tanh')(d4)
    d6 = Conv3D(nb_filters2, kernel_size, activation='tanh')(d5)

    r3 = AveragePooling3D(pool_size)(r2)
    r4 = Dropout(dropout_rate1)(r3)
    r5 = Conv3D(nb_filters2, kernel_size, padding='same', activation='tanh')(r4)
    r6 = Conv3D(nb_filters2, kernel_size, activation='tanh')(r5)
    g2 = Conv3D(1, (1, 1, 1), padding='same', activation='sigmoid')(r6)
    g2 = Attention_mask()(g2)
    gated2 = multiply([d6, g2])
    d7 = AveragePooling3D(pool_size)(gated2)
    d8 = Dropout(dropout_rate1)(d7)

    d9 = Flatten()(d8)
    d10_y = Dense(nb_dense, activation='tanh')(d9)
    d11_y = Dropout(dropout_rate2)(d10_y)
    out_y = Dense(n_frame, name='output_1')(d11_y)

    d10_r = Dense(nb_dense, activation='tanh')(d9)
    d11_r = Dropout(dropout_rate2)(d10_r)
    out_r = Dense(n_frame, name='output_2')(d11_r)

    model = Model(inputs=[diff_input, rawf_input], outputs=out_r)

    return model

#training_generator = DataGenerator('/home/ouzar1/Documents/pythonProject/BP4D-Expressive-Segment/BP4D-Expressive-Segment', 50, (120, 160), batch_size=1, temporal = 'MTTS_CAN')


# %% Create data genener
# training_generator = DataGenerator('/home/ouzar1/Documents/pythonProject/BP4D-Expressive-Segment/BP4D-Expressive-Segment', 10, (160, 120),
#                                    batch_size=1, frame_depth=10,
#                                    temporal= 'MTTS_CAN')
# validation_generator = DataGenerator('/home/ouzar1/Documents/pythonProject/BP4D-Expressive-Segment/BP4D-Expressive-Segment', 10, (160, 120),
#                                    batch_size=1, frame_depth=10,
#                                    temporal= 'MTTS_CAN')

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


def Custom():
    # Determine proper input shape

    diff_input = Input(shape=(50, 160, 120, 3))
    raw_input = Input(shape=(50, 160, 120, 3))

    # Block 1
    x = Conv3D(32, (3, 3, 3), strides=(1, 2, 2), padding='same', kernel_regularizer=l1_l2(l1=0.001, l2=0.001),
               use_bias=True)(
        raw_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv3D(32, (3, 3, 3), strides=(1, 2, 2), padding='same', kernel_regularizer=l1_l2(l1=0.001, l2=0.001),
               use_bias=True)(
        x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    y = Conv3D(32, (3, 3, 3), strides=(1, 2, 2), padding='same', kernel_regularizer=l1_l2(l1=0.001, l2=0.001),
               use_bias=True)(
        diff_input)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = Conv3D(32, (3, 3, 3), strides=(1, 2, 2), padding='same', kernel_regularizer=l1_l2(l1=0.001, l2=0.001),
               use_bias=True)(
        y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    att1 = x

    att1 = Conv3D(1, (1, 1, 1), padding='same', kernel_regularizer=l1_l2(l1=0.001, l2=0.001),
                  use_bias=False)(att1)

    att1 = BatchNormalization()(att1)

    # att1 = se_block(att1, num_filters=1)
    att1 = Attention_mask()(att1)

    y1 = multiply([y, att1])
    y1 = AveragePooling3D((3, 3, 3), strides=(1, 2, 2), padding='same')(y1)
    y1 = Dropout(0.2)(y1)

    x1 = AveragePooling3D((3, 3, 3), strides=(1, 2, 2), padding='same')(x)
    x1 = Dropout(0.2)(x1)

    x1 = Conv3D(64, (3, 3, 3), strides=(1, 2, 2), padding='same', kernel_regularizer=l1_l2(l1=0.001, l2=0.001),
                use_bias=True)(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    x1 = Conv3D(64, (3, 3, 3), strides=(1, 2, 2), padding='same', kernel_regularizer=l1_l2(l1=0.001, l2=0.001),
                use_bias=True)(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)

    y2 = Conv3D(64, (3, 3, 3), strides=(1, 2, 2), padding='same', kernel_regularizer=l1_l2(l1=0.001, l2=0.001),
                use_bias=True)(
        y1)
    y2 = BatchNormalization()(y2)
    y2 = Activation('relu')(y2)
    y2 = Conv3D(64, (3, 3, 3), strides=(1, 2, 2), padding='same', kernel_regularizer=l1_l2(l1=0.001, l2=0.001),
                use_bias=True)(
        y2)
    y2 = BatchNormalization()(y2)
    y2 = Activation('relu')(y2)
    # sum = add([x, att1])
    # relu2 = Activation('relu')(sum)

    att2 = x1

    att2 = Conv3D(1, (1, 1, 1), padding='same', kernel_regularizer=l1_l2(l1=0.001, l2=0.001),
                  use_bias=False)(att2)

    att2 = BatchNormalization()(att2)

    att2 = Attention_mask()(att2)

    y2 = multiply([y2, att2])

    # att2 = se_block(att2, num_filters=1)

    # y2 = multiply([att2, y1])
    # y2 = AveragePooling3D((3, 3, 3), strides=(2, 2, 2), padding='same')(y2)
    # y2 = Dropout(0.2)(y2)

    # sum = add([x, att2])
    # relu2 = Activation('relu')(sum)

    y3 = AveragePooling3D((3, 3, 3), strides=(1, 5, 4), padding='same')(y2)
    y3 = Dropout(0.5)(y3)

    y3 = Dense(128, kernel_regularizer=l1_l2(l1=0.001, l2=0.001), activation='relu')(y3)
    y3 = Dropout(0.5)(y3)
    y3 = Dense(50, kernel_regularizer=l1_l2(l1=0.001, l2=0.001), activation='linear')(y3)
    # model = Model(inputs=[diff_input, raw_input], outputs=y1)

    model = Model(inputs=[diff_input, raw_input], outputs=y3)

    # model = Model(inputs, x, name='xception')
    return model


model = Custom()
model.summary()

input_shape = (160, 120, 3)

# model = MT_CAN_3D(50, 32, 64, input_shape, kernel_size=(3, 3, 3), dropout_rate1=0.25,
#               dropout_rate2=0.5, pool_size=(2, 2, 2), nb_dense=128)
opt = RAdam(learning_rate=0.0001, decay=0.01)
rmse = RootMeanSquaredError()

model.compile(loss='mse', optimizer=opt, metrics=['mae', rmse])

from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import CSVLogger

# every 10 epochs save weights
checkpoint = ModelCheckpoint('weights.{epoch:02d}-{val_root_mean_squared_error:.4f}.h5',
                             monitor='val_loss',
                             verbose=10, save_best_only=False)
history_checkpoint = CSVLogger("hist_CAN.csv", append=True)

# %% Model Training and Saving Results
history = model.fit(train_generator, epochs=100,
                                  steps_per_epoch= len(train.filenames) // 50,
                                  validation_data=test_generator, validation_steps=len(test.filenames) // 50, callbacks=[history_checkpoint, checkpoint])



def Xception():

	# Determine proper input shape
    vid_input = Input(shape=(100, 120, 160, 3))
    PPG_input= Input(shape=(100,))

    # Block 1
    x = Conv3D(8, (3, 3, 3), strides=(2, 2, 2), kernel_regularizer=l1_l2(l1=0.001, l2=0.001), use_bias=False)(vid_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv3D(16, (3, 3, 3), kernel_regularizer=l1_l2(l1=0.001, l2=0.001), use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    residual = Conv3D(32, (1, 1, 1), strides=(2, 2, 2), padding='same', kernel_regularizer=l1_l2(l1=0.001, l2=0.001),
                      use_bias=False)(x)
    residual = BatchNormalization()(residual)

    # Block 2
    x = SeparableConv3D(kernel_size=(3, 3, 3), depth_multiplier=1)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv3D(kernel_size=(3, 3, 3), depth_multiplier=2)(x)
    x = BatchNormalization()(x)

    # Block 2 Pool
    x = AveragePooling3D((3, 3, 3), strides=(2, 2, 2), padding='same')(x)
    x = add([x, residual])

    residual = Conv3D(64, (1, 1, 1), strides=(2, 2, 2), padding='same', kernel_regularizer=l1_l2(l1=0.001, l2=0.001),
                      use_bias=False)(x)
    residual = BatchNormalization()(residual)

    # Block 3
    x = Activation('relu')(x)
    x = SeparableConv3D(kernel_size=(3, 3, 3), depth_multiplier=2)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv3D(kernel_size=(3, 3, 3), depth_multiplier=1)(x)
    x = BatchNormalization()(x)

    # Block 3 Pool
    x = AveragePooling3D((3, 3, 3), strides=(2, 2, 2), padding='same')(x)

    x = add([x, residual])

    residual = Conv3D(256, (1, 1, 1), strides=(2, 2, 2), padding='same', kernel_regularizer=l1_l2(l1=0.001, l2=0.001),
                      use_bias=False)(x)
    residual = BatchNormalization()(residual)

    # Block 4
    x = Activation('relu')(x)
    x = SeparableConv3D(kernel_size=(3, 3, 3), depth_multiplier=2)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv3D(kernel_size=(3, 3, 3), depth_multiplier=2)(x)
    x = BatchNormalization()(x)

    # Block 4 Pool
    x = AveragePooling3D((3, 3, 3), strides=(2, 2, 2), padding='same')(x)

    x = add([x, residual])

    # Block 5 - 12
    for i in range(8):
        residual = x

        x = Activation('relu')(x)
        x = SeparableConv3D(kernel_size=(3, 3, 3), depth_multiplier=1)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = SeparableConv3D(kernel_size=(3, 3, 3), depth_multiplier=1)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = SeparableConv3D(kernel_size=(3, 3, 3), depth_multiplier=1)(x)
        x = BatchNormalization()(x)
        x = add([x, residual])

    residual = Conv3D(256, (1, 1, 1), strides=(2, 2, 2), padding='same', kernel_regularizer=l1_l2(l1=0.001, l2=0.001),
                      use_bias=False)(x)
    residual = BatchNormalization()(residual)

    # Block 13
    x = Activation('relu')(x)
    x = SeparableConv3D(kernel_size=(3, 3, 3), depth_multiplier=1)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv3D(kernel_size=(3, 3, 3), depth_multiplier=1)(x)
    x = BatchNormalization()(x)

    # Block 13 Pool
    x = AveragePooling3D((3, 3, 3), strides=(2, 2, 2), padding='same')(x)

    x = add([x, residual])

    # Block 14
    x = SeparableConv3D(kernel_size=(3, 3, 3), depth_multiplier=1)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Block 14 part 2
    x = SeparableConv3D(kernel_size=(3, 3, 3), depth_multiplier=2)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Fully Connected Layer
    x = Flatten()(x)
    x = Dense(1024, kernel_regularizer=l1_l2(l1=0.001, l2=0.001), activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Model(inputs=vid_input, outputs=x)

    # the second branch opreates on the second input
    y = Dense(100, activation="relu")(PPG_input)
    y = Dense(64, activation="relu")(y)

    y = Model(inputs=PPG_input, outputs=y)

    # combine the output of the two branches
    combined = concatenate([x.output, y.output])
    # apply a FC layer and then a regression prediction on the
    # combined outputs
    z = Dense(32, activation="relu")(combined)
    z = Dropout(0.1)(z)
    z = Dense(4, kernel_regularizer=l1_l2(l1=0.001, l2=0.001), activation='softmax')(z)
    # our model will accept the inputs of the two branches and
    # then output a single value
    #model = Model(inputs=[x.input, y.input], outputs=z)
    model = Model([vid_input, PPG_input], z, name='xception')
    return model


#
# model =Xception()
# epochs = 25
# drop_rate = 0.1
# lr = 0.01
# #model = densenet_3d(1, input_shape, dropout_rate=drop_rate)
# #model = resnet(input_shape)
# #model = CNNModel()
#
#
# opt = RAdam(learning_rate=0.0001, decay=0.01)
# rmse = RootMeanSquaredError()
#
# model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
# #model.compile(optimizer = Adam(learning_rate=0.001), loss = 'categorical_crossentropy', metrics = ["accuracy"])
# model.summary()
#
# print('Start training..')
#
# from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
# from tensorflow.keras.callbacks import EarlyStopping
# from tensorflow.keras.callbacks import CSVLogger
# import pandas as pd
#
# # every epoch check validation accuracy scores and save the highest
# checkpoint_2 = ModelCheckpoint('weights-{epoch:02d}.h5',
#                                monitor='val_root_mean_squared_error',
#                                verbose=1, save_best_only=False)
# # every 10 epochs save weights
# checkpoint = ModelCheckpoint('weights.{epoch:02d}-{val_accuracy:.4f}.h5',
#                              monitor='val_accuracy',
#                              verbose=10, save_best_only=True)
# history_checkpoint = CSVLogger("hist_multimodal.csv", append=True)
#
# # use tensorboard can watch the change in time
# tensorboard_ = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
#
# early_stopping = EarlyStopping(monitor='val_root_mean_squared_error', patience=5, verbose=1, mode='auto')
# """
# if (CONTINUE_TRAINING == True):
#     history = pd.read_csv('history2.csv')
#     INITIAL_EPOCH = history.shape[0]
#     model.load_weights('weights_%02d.h5' % INITIAL_EPOCH)
#     checkpoint_2.best = np.min(history['val_root_mean_squared_error'])
# else:
#     INITIAL_EPOCH = 0
# """
#
#
#
# history = model.fit(train_generator, epochs=50,
#                                   steps_per_epoch= len(train_data.filenames) // 100,
#                                   validation_data=test_generator , validation_steps=len(test_data.filenames) // 100, callbacks=[history_checkpoint, checkpoint], use_multiprocessing=True)
#
# values = history.history
#
# validation_loss = values['val_loss']
# validation_accuracy = values['val_accuracy']
# training_accuracy = values['accuracy']
# training_loss = values['loss']
#
# epochs = range(50)
#
# plt.plot(epochs, training_loss, label='Training Loss')
# plt.plot(epochs, validation_loss, label='Validation Loss')
# plt.title('Epochs vs Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()
#
# plt.plot(epochs, training_accuracy, label='Training accuracy')
# plt.plot(epochs, validation_accuracy, label='Validation accuracy')
# plt.title('Epochs vs accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('accuracy')
# plt.legend()
# plt.show()
#
# scores = model.evaluate(test_generator, len(test_data.filenames) // 100)
# #scores = model.predict_generator(train_data, len(train_data.filenames) // 200)
# #print("%s: %f" % (model.metrics_names[1], scores[1]))
# print(scores)
# # scores = model.evaluate(train_data) ,kernel_regularizer=l2(0.001)
#
#
#
