import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import tensorflow as tf

# Disable eager execution (Adam optimizer cannot be used if this option is enabled)
tf.compat.v1.disable_eager_execution()
tf.executing_eagerly()

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, BatchNormalization, Activation, Flatten, Dropout, AveragePooling3D, Conv3D, add, concatenate, multiply, Reshape, GlobalAveragePooling3D
from tensorflow.keras.metrics import  RootMeanSquaredError
from keras_radam import RAdam
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.optimizers import Adam
from DepthwiseConv3D1 import DepthwiseConv3D
SeparableConv3D = DepthwiseConv3D

batch_size_train = 26
batch_size_test = 1
import matplotlib.pyplot as plt

from Generator_FER import ImageDataGenerator
datagen = ImageDataGenerator()

# train_data = datagen.flow_from_directory(directory='/home/ouzar1/Documents/pythonProject/BP4D-Expressive-Segment/multimodal/FE/Train_data',
#                                              target_size=(160, 160), class_mode='categorical', batch_size=batch_size_train,
#                                              frames_per_step=100, shuffle=False)
#
# test_data = datagen.flow_from_directory(directory='/home/ouzar1/Documents/pythonProject/BP4D-Expressive-Segment/BP4D-Expressive-Segment/Test_set',
#                                              target_size=(160, 120), class_mode='categorical', batch_size=batch_size_test,
#                                              frames_per_step=100, shuffle=False)

from HRV_Features_Generator import ImageDataGenerator
datagen1 = ImageDataGenerator()

def train_generator_multiple():
    genX1 = datagen.flow_from_directory(directory='/home/ouzar1/Documents/pythonProject/BP4D-Expressive-Segment/multimodal/FE/Train_data',
                                             target_size=(160, 120), class_mode='categorical', batch_size=batch_size_train,
                                             frames_per_step=100, shuffle=False)

    genX2 = datagen1.flow_from_directory(directory='/home/ouzar1/Documents/pythonProject/BP4D-Expressive-Segment/multimodal/FE/Train_data',
                                              label_dir='/home/ouzar1/Documents/pythonProject/BP4D-Expressive-Segment/multimodal/PPG/Train_data',
                                             target_size=(160, 120), class_mode='HRV', batch_size=batch_size_train,
                                             frames_per_step=100, shuffle=False)

    while True:
        X1i = genX1.next()
        X2i = genX2.next()
        yield [X1i[0], X2i[1]], X1i[1]  # Yield both images and their mutual label

def test_generator_multiple():
    genX1 = datagen.flow_from_directory(directory='/home/ouzar1/Documents/pythonProject/BP4D-Expressive-Segment/BP4D-Expressive-Segment/Test_set',
                                             target_size=(160, 120), class_mode='categorical', batch_size=batch_size_test,
                                             frames_per_step=100, shuffle=False)

    genX2 = datagen1.flow_from_directory(directory='/home/ouzar1/Documents/pythonProject/BP4D-Expressive-Segment/BP4D-Expressive-Segment/Test_set',
                                              label_dir='/home/ouzar1/Documents/pythonProject/BP4D-Expressive-Segment/multimodal/PPG/Test_data',
                                             target_size=(160, 120), class_mode='HRV', batch_size=batch_size_test,
                                             frames_per_step=100, shuffle=False)

    while True:
        X1i = genX1.next()
        X2i = genX2.next()
        yield [X1i[0], X2i[1]], X1i[1]  # Yield both images and their mutual label

train_generator = train_generator_multiple()
test_generator = test_generator_multiple()

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
    y = Dense(23, activation="relu")(HRV_input)
    # y = Dense(64, activation="relu")(y)

    y = Model(inputs=HRV_input, outputs=y)

    # combine the output of the two branches
    combined = concatenate([x.output, y.output])
    # apply a FC layer and then a regression prediction on the
    # combined outputs
    z = Dense(256, activation="relu")(combined)
    z = Dropout(0.1)(z)
    z = Dense(4, kernel_regularizer=l1_l2(l1=0.001, l2=0.001), activation='softmax')(z)
    # our model will accept the inputs of the two branches and
    # then output a single value
    #model = Model(inputs=[x.input, y.input], outputs=z)
    model = Model([vid_input, HRV_input], z, name='xception')
    return model


def SE_Xception():

	# Determine proper input shape
	#input_shape = _obtain_input_shape(None, default_size=299, min_size=71, data_format='channels_last', include_top=False)

    img_input = Input(shape=(100, 160, 120, 3))
    HRV_input = Input(shape=(23,))

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
    x = Flatten()(d12)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.4)(x)
    x = Model(inputs=img_input, outputs=x)

    # the second branch opreates on the second input
    y = Dense(23, activation="relu")(HRV_input)
    y = Model(inputs=HRV_input, outputs=y)

    # combine the output of the two branches
    combined = concatenate([x.output, y.output])

    # apply a FC layer and then a regression prediction on the
    # combined outputs
    z = Dense(128, activation="relu")(combined)
    z = Dropout(0.4)(z)
    z = Dense(4, kernel_regularizer=l1_l2(l1=0.001, l2=0.001), activation='softmax')(z)
    # our model will accept the inputs of the two branches and
    # then output a single value
    # model = Model(inputs=[x.input, y.input], outputs=z)
    model = Model([img_input, HRV_input], z, name='SE-xception')
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



history = model.fit(train_generator, epochs=50,
                                  steps_per_epoch= 226,
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

scores = model.evaluate(test_generator, len(test_data.filenames) // 100)
#scores = model.predict_generator(train_data, len(train_data.filenames) // 200)
#print("%s: %f" % (model.metrics_names[1], scores[1]))
print(scores)
# scores = model.evaluate(train_data) ,kernel_regularizer=l2(0.001)



