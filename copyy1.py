import os
from shutil import copyfile

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import math
import numpy as np
import cv2
# from numpy import *
import matplotlib.pyplot as plt
import csv
import pandas as pd
# from PIL import Image
# from sklearn import preprocessing
# from keras.preprocessing.image import load_img
# from keras.preprocessing.image import img_to_array
# from keras.preprocessing.image import array_to_img
# from mtcnn import MTCNN
# import keras
# from keras.models import Sequential
# from keras.layers.core import Activation, Flatten, Dense, Dropout
# from keras.layers import Activation, Dense
# from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D, Conv3D, MaxPooling3D, AveragePooling2D
# from keras.optimizers import SGD
#import cv2
#from imutils import face_utils
#import dlib
# from tensorflow.python.keras.utils import np_utils
# from keras import backend as K
# K.common.image_dim_ordering()
# K.set_image_dim_ordering('th')
#from keras.layers import *
#from keras.layers.advanced_activations import LeakyReLU
#from keras.activations import relu
#from keras.initializers import RandomNormal
#from keras.applications import *
#import keras.backend as K
#from face_detector import *
#from detection_viewer import DetectionViewer
#from image_cropper import ImageCropper
#import time
#from FCN8s_keras import FCN

#model = FCN()
#model.load_weights("Keras_FCN8s_face_seg_YuvalNirkin.h5")

def get_im_train(path_im, path_save_im, path_hr):
    global data_train
    list_dir = sorted(os.listdir(path_im))
    #print(sorted(list_dir))
    count = 0
    file_count = 0
    data_train = []
    train_data = []
    train_data1 = []

    global image1
    image1 = []
    for i in range(int(len(list_dir))):
    #for i in range(100, 110):
        list_dir1 = sorted(os.listdir(path_im + '/' +  list_dir[i]))
        list_dir_save1 = path_save_im + '/' +  list_dir[i]
        dir_hr = path_hr + '/' +  list_dir[i]
        list_dir_hr = sorted(os.listdir(dir_hr))
        #print(list_dir_hr)
        #if not os.path.exists(list_dir_save1):
            #os.makedirs(list_dir_save1)
        Heart_rate_dir1=[]
        #for j in range(0,6):

        for j in range(int(len(list_dir1))):
            path_to_files=path_im + '/' +  list_dir[i] + '/' + list_dir1[j]
            list_dir2 = os.listdir(path_to_files)
            list_dir_save2 = path_save_im + '/' + list_dir[i] + '/' + list_dir1[j]
            if not os.path.exists(list_dir_save2):
                os.makedirs(list_dir_save2)
            #for hr in sorted(list_dir_hr):
            pulse_rate = [filename for filename in list_dir_hr if filename.endswith(".txt")]


            for pr in pulse_rate:
                pr1 = os.path.join(path_hr + '/' + list_dir[i] + '/' + pr)

                path_to_save = path_save_im + '/' + list_dir[i] + '/' + list_dir1[j] + '/' + pr
                #pr2 = os.rename(pr1, path_hr + '/' + list_dir[i] + '/' + 'Pulse_rate.txt')
                print(pr1, path_to_save)
                copyfile(pr1, path_to_save)

            #print(path_to_files, list_dir_save2)
            #print(list_dir2)
            #if not os.path.exists(list_dir_save2):
                #os.makedirs(list_dir_save2)
                #copyfile(imag, imag1)

path_im = '/home/ouzar1/Desktop/Dataset1/mahnob1/ROI1'
path_hr = '/media/bousefsa1/Elements/BD PPG/2 bases publiques/manhob hci/Sessions'

path_save_im = '/home/ouzar1/Desktop/Dataset1/mahnob1/HR1'

print("begin")
get_im_train(path_im, path_save_im, path_hr)
print("finished")

