import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import tensorflow as tf

# Disable eager execution (Adam optimizer cannot be used if this option is enabled)
tf.compat.v1.disable_eager_execution()
tf.executing_eagerly()

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, BatchNormalization, Activation, Flatten, Dropout, AveragePooling3D, Conv3D, add, concatenate, multiply, Reshape, GlobalAveragePooling3D, MaxPooling3D, GlobalAveragePooling2D
from tensorflow.keras.layers import LSTM
from tensorflow.keras.metrics import  RootMeanSquaredError
from keras_radam import RAdam
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.optimizers import Adam
from DepthwiseConv3D1 import DepthwiseConv3D
SeparableConv3D = DepthwiseConv3D

batch_size_train = 4
batch_size_test = 1
import matplotlib.pyplot as plt


#for data in train_generator:
    #image = data[0]
    #label = data[1]
    #print(image[0].shape, image[1].shape, label.shape)

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
    vid_input = Input(shape=(100, 160, 120, 3))
    HRV_input= Input(shape=(23,))

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
    y = Dense(100, activation="relu")(HRV_input)
    y = Dense(64, activation="relu")(y)

    y = Model(inputs=HRV_input, outputs=y)

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
    model = Model([vid_input, HRV_input], z, name='xception')
    return model


def Xception():

	# Determine proper input shape
    vid_input = Input(shape=(100, 160, 120, 3))
    landmarks_input= Input(shape=(100,30,2))

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
    # x = Dropout(0.1)(x)
    x = Model(inputs=vid_input, outputs=x)

    # the second branch opreates on the second input

    # y = Flatten()(landmarks_input)
    y = GlobalAveragePooling2D()(landmarks_input)

    y = Dense(100, activation="relu")(y)
    # y = Dense(64, activation="relu")(y)

    y = Model(inputs=landmarks_input, outputs=y)

    # combine the output of the two branches
    combined = concatenate([x.output, y.output])
    # apply a FC layer and then a regression prediction on the
    # combined outputs
    z = Dense(128, activation="relu")(combined)
    # z = Dropout(0.1)(z)
    z = Dense(4, kernel_regularizer=l1_l2(l1=0.01, l2=0.01), activation='softmax')(z)
    # our model will accept the inputs of the two branches and
    # then output a single value
    #model = Model(inputs=[x.input, y.input], outputs=z)
    model = Model([vid_input, landmarks_input], z, name='xception')
    return model

def Xception():

	# Determine proper input shape
    vid_input = Input(shape=(100, 160, 120, 3))
    landmarks_input= Input(shape=(100,30,2))

    # Block 1
    x = Conv3D(64, (3, 3, 3), padding='same', kernel_regularizer=l1_l2(l1=0.01, l2=0.01), use_bias=False)(vid_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv3D(64, (3, 3, 3), padding='same', kernel_regularizer=l1_l2(l1=0.01, l2=0.01), use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling3D((3, 3, 3), strides=(1, 4, 3), padding='same')(x)

    # Block 2
    x = Conv3D(128, (3, 3, 3), padding='same', kernel_regularizer=l1_l2(l1=0.01, l2=0.01), use_bias=False)(vid_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv3D(128, (3, 3, 3), padding='same', kernel_regularizer=l1_l2(l1=0.01, l2=0.01), use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling3D((3, 3, 3), strides=(1, 2, 2), padding='same')(x)
    low = x
    # Block 3
    x = Conv3D(256, (3, 3, 3), padding='same', kernel_regularizer=l1_l2(l1=0.01, l2=0.01), use_bias=False)(vid_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv3D(256, (3, 3, 3), padding='same', kernel_regularizer=l1_l2(l1=0.01, l2=0.01), use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv3D(256, (3, 3, 3), padding='same', kernel_regularizer=l1_l2(l1=0.01, l2=0.01), use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling3D((3, 3, 3), strides=(1, 2, 2), padding='same')(x)

    # Block 4
    x = Conv3D(512, (3, 3, 3), padding='same', kernel_regularizer=l1_l2(l1=0.01, l2=0.01), use_bias=False)(vid_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv3D(512, (3, 3, 3), padding='same', kernel_regularizer=l1_l2(l1=0.01, l2=0.01), use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv3D(512, (3, 3, 3), padding='same', kernel_regularizer=l1_l2(l1=0.01, l2=0.01), use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling3D((3, 3, 3), strides=(1, 2, 2), padding='same')(x)

    # Block 5
    x = Conv3D(512, (3, 3, 3), padding='same', kernel_regularizer=l1_l2(l1=0.01, l2=0.01), use_bias=False)(vid_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv3D(512, (3, 3, 3), padding='same', kernel_regularizer=l1_l2(l1=0.01, l2=0.01), use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv3D(512, (3, 3, 3), padding='same', kernel_regularizer=l1_l2(l1=0.01, l2=0.01), use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling3D((3, 3, 3), strides=(1, 5, 5), padding='same')(x)
    mid = x

    # Fully Connected Layer
    x = Flatten()(x)
    # x = Dense(256, kernel_regularizer=l1_l2(l1=0.01, l2=0.01), activation='relu')(x)
    # x = Dropout(0.1)(x)
    x = Model(inputs=vid_input, outputs=x)

    y = GlobalAveragePooling2D()(landmarks_input)

    # y = Dense(100, activation="relu")(y)

    y = Model(inputs=landmarks_input, outputs=y)

    # combine the output of the two branches
    combined = concatenate([x.output, y.output])
    # apply a FC layer and then a regression prediction on the
    # combined outputs
    # z = Dense(128, activation="relu")(combined)

    combined1 = concatenate([low, mid])


    # z = Dropout(0.1)(z)
    z = Dense(4, kernel_regularizer=l1_l2(l1=0.01, l2=0.01), activation='softmax')(combined1)
    # our model will accept the inputs of the two branches and
    # then output a single value
    # model = Model(inputs=[x.input, y.input], outputs=z)
    model = Model([vid_input, landmarks_input], z, name='xception')
    return model


# model =SE_Xception()
model =Xception()

epochs = 25
drop_rate = 0.1
lr = 0.01
#model = densenet_3d(1, input_shape, dropout_rate=drop_rate)
#model = resnet(input_shape)
#model = CNNModel()


opt = RAdam(learning_rate=0.0001, decay=0.01)
rmse = RootMeanSquaredError()

model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
#model.compile(optimizer = Adam(learning_rate=0.001), loss = 'categorical_crossentropy', metrics = ["accuracy"])
model.summary()

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
                             verbose=10, save_best_only=True)
history_checkpoint = CSVLogger("hist_multimodal.csv", append=True)

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



from Generator_FER import ImageDataGenerator
datagen = ImageDataGenerator()

# train_data = datagen.flow_from_directory(directory='/home/ouzar1/Documents/pythonProject/BP4D-Expressive-Segment/multimodal/FE/Train_data',
#                                              target_size=(160, 160), class_mode='categorical', batch_size=batch_size_train,
#                                              frames_per_step=100, shuffle=False)
#
# test_data = datagen.flow_from_directory(directory='/home/ouzar1/Documents/pythonProject/BP4D-Expressive-Segment/BP4D-Expressive-Segment/Test_set',
#                                              target_size=(160, 120), class_mode='categorical', batch_size=batch_size_test,
#                                              frames_per_step=100, shuffle=False)

from Generator_landmarks import ImageDataGenerator
datagen1 = ImageDataGenerator()

def train_generator_multiple():
    genX1 = datagen.flow_from_directory(directory='/home/ouzar1/Documents/pythonProject/BP4D-Expressive-Segment/multimodal/FE/Train_data',
                                             target_size=(160, 120), class_mode='categorical', batch_size=batch_size_train,
                                             frames_per_step=100, shuffle=False)

    genX2 = datagen1.flow_from_directory(directory='/home/ouzar1/Documents/pythonProject/BP4D-Expressive-Segment/multimodal/FE/Train_data',
                                             target_size=(160, 120), class_mode='categorical', batch_size=batch_size_train,
                                             frames_per_step=100, shuffle=False)

    while True:
        X1i = genX1.next()
        X2i = genX2.next()
        yield [X1i[0], X2i[0]], X1i[1]  # Yield both images and their mutual label

def test_generator_multiple():
    genX1 = datagen.flow_from_directory(directory='/home/ouzar1/Documents/pythonProject/BP4D-Expressive-Segment/BP4D-Expressive-Segment/Test_set',
                                             target_size=(160, 120), class_mode='categorical', batch_size=batch_size_test,
                                             frames_per_step=100, shuffle=False)

    genX2 = datagen1.flow_from_directory(directory='/home/ouzar1/Documents/pythonProject/BP4D-Expressive-Segment/BP4D-Expressive-Segment/Test_set',
                                             target_size=(160, 120), class_mode='categorical', batch_size=batch_size_test,
                                             frames_per_step=100, shuffle=False)

    while True:
        X1i = genX1.next()
        X2i = genX2.next()
        yield [X1i[0], X2i[0]], X1i[1]  # Yield both images and their mutual label

train_generator = train_generator_multiple()
test_generator = test_generator_multiple()


history = model.fit(train_generator, epochs=50,
                                  steps_per_epoch= 1469,
                                  validation_data=test_generator , validation_steps= 153, callbacks=[history_checkpoint, checkpoint], use_multiprocessing=False)

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

