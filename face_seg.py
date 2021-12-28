import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
from scipy.spatial import ConvexHull
import skimage
from skimage.draw import polygon
#from numpy import *
import matplotlib.pyplot as plt
import csv
import pandas as pd
#from PIL import Image
#from sklearn import preprocessing
#from keras.preprocessing.image import load_img
#from keras.preprocessing.image import img_to_array
#from keras.preprocessing.image import array_to_img
#from mtcnn import MTCNN
#import keras
#from keras.models import Sequential
#from keras.layers.core import Activation, Flatten, Dense, Dropout
#from keras.layers import Activation, Dense
#from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D, Conv3D, MaxPooling3D, AveragePooling2D
#from keras.optimizers import SGD
import cv2
from imutils import face_utils
import dlib
#from tensorflow.python.keras.utils import np_utils
#from keras import backend as K
#K.common.image_dim_ordering()
#K.set_image_dim_ordering('th')
from keras.layers import *
from keras.layers.advanced_activations import LeakyReLU
from keras.activations import relu
from keras.initializers import RandomNormal
from keras.applications import *
import keras.backend as K
from face_detector import *
from detection_viewer import DetectionViewer
from image_cropper import ImageCropper
import time
from FCN8s_keras import FCN

model = FCN()
model.load_weights("Keras_FCN8s_face_seg_YuvalNirkin.h5")


def vgg_preprocess(im):
    im = cv2.resize(im, (500, 500))
    in_ = np.array(im, dtype=np.float32)
    in_ = in_[:, :, ::-1]
    in_ -= np.array((104.00698793, 116.66876762, 122.67891434))
    in_ = in_[np.newaxis, :]
    # in_ = in_.transpose((2,0,1))
    return in_
def auto_downscaling(im):
    w = im.shape[1]
    h = im.shape[0]
    while w * h >= 700 * 700:
        im = cv2.resize(im, (0, 0), fx=0.5, fy=0.5)
        w = im.shape[1]
        h = im.shape[0]
    return im

def roi_seg(image):
    im = auto_downscaling(image)
    # vgg_preprocess: output BGR channel w/ mean substracted.
    inp_im = vgg_preprocess(im)
    out = model.predict([inp_im])
    # post-process for display
    out_resized = cv2.resize(np.squeeze(out), (im.shape[1], im.shape[0]))
    out_resized_clipped = np.clip(out_resized.argmax(axis=2), 0, 1).astype(np.float64)

    mask = cv2.GaussianBlur(out_resized_clipped, (7, 7), 6)
    # plt.imshow((mask[:,:,np.newaxis]*im.astype(np.float64)).astype(np.uint8))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

    #cv2.imshow('img', (mask[:, :, np.newaxis] * im.astype(np.float64)).astype(np.uint8))
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    #cv2.imwrite('D:/roi_a_verifier/1/T11/face_seg.jpg', (mask[:, :, np.newaxis] * im.astype(np.float64)).astype(np.uint8))
    image = (mask[:, :, np.newaxis] * im.astype(np.float64)).astype(np.uint8)
    #image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    #cv2.imshow('img', image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    #image1 = np.resize(image,(110,260,3))
    #print(image1.shape, image1.ndim)
    return image

# https://towardsdatascience.com/facial-mapping-landmarks-with-dlib-python-160abcf7d672
def draw_convex_hull(im, points, color):
    points = cv2.convexHull(points)
    cv2.fillConvexPoly(im, points, color=color)


# from get_roi import get_roi
def mask_roi(image, bbox):
    p = '/media/bousefsa1/Seagate Backup Plus Drive/Face_segmentation/shape_predictor_81_face_landmarks.dat'
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(p)

    #bbox = dlib_detector.detect()
    #dlib_detector = DlibCVFaceDetector()
    #mtcnn_detector = MTCNNFaceDetector()


    if detector != []:

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        rects = detector(gray, 0)
        if (rects is None):
            return ()
        result = list(enumerate(rects))
        # For each detected face, find the landmark.
        if result != []:
            for (i, rect) in enumerate(rects):
                # Make the prediction and transfom it to numpy array
                shape = predictor(gray, rect)
                #shape = face_utils.shape_to_np(shape)
                landmarks = np.array([[p.x, p.y] for p in shape.parts()])

                outline1 = landmarks[[*range(17, 27), 16, 74, 29, 35, 33, 31, 29, 0, 75]]
                outline2 = landmarks[[30, 35, *range(48, 61), 31]]
                outline3 = landmarks[[48, *range(4, 13), 54]]
                outline4 = landmarks[[17, 4, 3, 2, 1, 0]]
                outline5 = landmarks[[26, 12, 13, 14, 15, 16]]
                outline5 = landmarks[[*range(17, 27), 78, 29, 35, 33, 31, 29, 77]]

                draw_convex_hull(image, outline5, color=0)
                draw_convex_hull(image, outline2, color=0)
                # draw_convex_hull(image, pts1, color=0)
                # draw_convex_hull(image, pts, color=0)
                #image = image[rect.top():rect.bottom(), rect.left():rect.right()]
        

                #draw_convex_hull(image, pts1, color=0)
                #draw_convex_hull(image, pts, color=0)

        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                             
        #img_crop = ImageCropper(image)
        #im = img_crop.crop(bbox)

        #im = cv2.resize(im,(160,240))

        #print(im.shape)
        return image
        #im = cv2.resize(im,(120,120))
        #print(im.shape)
        #return im



def get_roi1(image):
    detector = MTCNN()
    global image1

    result = detector.detect_faces(image)
    if result != []:
        for person in result:
            bounding_box = person['box']
            keypoints = person['keypoints']

    #detector.detect_faces(image)

    #result = detector.detect_faces(image)

    # Result is an array with all the bounding boxes detected. We know that for 'ivan.jpg' there is only one.
    #bounding_box = result[0]['box']

    #keypoints = result[0]['keypoints']

            cv2.circle(image,(keypoints['left_eye']), 2, (0,155,255), 2)
            cv2.circle(image,(keypoints['right_eye']), 2, (0,155,255), 2)
            cv2.circle(image,(keypoints['nose']), 2, (0,155,255), 2)
            cv2.circle(image,(keypoints['mouth_left']), 2, (0,155,255), 2)
            cv2.circle(image,(keypoints['mouth_right']), 2, (0,155,255), 2)

            image = image[bounding_box[1] : bounding_box[1] + bounding_box[3] , bounding_box[0] : bounding_box[0] + bounding_box[2]]

            left_eye=[]
            left_eye = keypoints['left_eye']
            a1=left_eye[0]
            a2=left_eye[1]
            x1 = a1 - bounding_box[0]

            right_eye=[]
            right_eye = keypoints['right_eye']
            b1=right_eye[0]
            x2 = b1 - bounding_box[0]
            b2=right_eye[1]

            y2= int((a2 - bounding_box[1])/2)

            #cv2.rectangle(image, (x1, 20), (x2, y2) , (0,155,255), 2)

            #image = image[20 : y2 , x1 : x2]
            #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            #image = img_to_array(image)
            #image = cv2.resize(image,(200,100))
            #print(image.shape)
            cv2.imshow('img',image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            #image1 = np.resize(image,(110,260,3))
            #print(image1.shape, image1.ndim)
    return image
#from get_roi import get_roi


def get_im_train(path_im, path_save_im):
    global data_train
    list_dir = os.listdir(path_im)
    count=0
    file_count=0
    data_train=[]
    train_data = []
    train_data1 = []
    global image1
    image1 = []
    for i in range(int(len(list_dir))):
        list_dir1 = os.listdir(path_im + '/' +  list_dir[i])
        list_dir_save1 = path_save_im + '/' +  list_dir[i]
        if not os.path.exists(list_dir_save1):
            os.makedirs(list_dir_save1)

        Heart_rate_dir1=[]
        for j in range(int(len(list_dir1))):
            path_to_files=path_im + '/' +  list_dir[i] + '/' + list_dir1[j]
            list_dir2 = os.listdir(path_to_files)
            for k in range(int(len(list_dir2))):
                path_to_files1 = path_im + '/' + list_dir[i] + '/' + list_dir1[j] + '/' + list_dir2[k]
                list_dir3 = os.listdir(path_to_files1)
                list_dir_save2 = path_save_im + '/' +  list_dir[i] + '/' + list_dir2[k]
                if not os.path.exists(list_dir_save2):
                    os.makedirs(list_dir_save2)

                for im in list_dir3:

                    imag = os.path.join(path_to_files1, im)
                    imag1 = os.path.join(list_dir_save2, im)


                #data_train.append(imag)
                    img=cv2.imread(imag)
                    #img = cv2.cvtColor(cv2.imread(imag), cv2.COLOR_RGB2BGR)
                    bgr_img = cv2.imread(imag)
                    img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (520, 696))

                #img = cv2.cvtColor(cv2.imread(imag), cv2.COLOR_BGR2RGB)
                #img = get_roi1(img)

                    mtcnn_detector = MTCNNFaceDetector()
                    bbox = mtcnn_detector.detect(img)
                    #image = image[rect.top():rect.bottom(), rect.left():rect.right()]

                    res = list(enumerate(bbox))
                # For each detected face, find the landmark.

                    if res != []:
                        img = roi_seg(img)
                        #roi = mask_roi(img,bbox)
                        img = img[ bbox[0][0] - 50: bbox[0][1] + 50 , bbox[0][2]  - 50: bbox[0][3] + 50]
                        #img = cv2.resize(img, (260, 360))


                    else :
                        print(imag)

                    #img_crop = ImageCropper(roi)
                    #im = img_crop.crop(bbox)
                    print(img.shape)
                    cv2.imshow('img', img)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                    #cv2.imwrite(imag1,img)
                    count += 1
                    print(count)
                #height = np.size(img, 0)
                #width = np.size(img, 1)
                #train_data.append(np.array(img))
                #train_data1 = np.array(train_data)
                #print(train_data1.shape)
    #return train_data1

path_im = '/media/bousefsa1/Seagate Backup Plus Drive/signal_test'
path_save_im = '/media/bousefsa1/Seagate Backup Plus Drive/ROI/Face_seg_with_mask_crop_manuallly'
print("begin")
get_im_train(path_im, path_save_im)
print("finished")


"""
        Heart_rate_dir1=[]
        for j in range(int(len(list_dir1))):
            path_to_files=path_im + '/' +  list_dir[i] + '/' + list_dir1[j]
            list_dir2 = os.listdir(path_to_files)
            for k in range(int(len(list_dir2))):
                path_to_files1 = path_im + '/' + list_dir[i] + '/' + list_dir1[j] + '/' + list_dir2[k]
                list_dir3 = os.listdir(path_to_files1)
                for im in list_dir3:

                    imag = os.path.join(path_to_files1, im)
                    #print(imag)


                #data_train.append(imag)
                    img=cv2.imread(imag)
                    #img = cv2.cvtColor(cv2.imread(imag), cv2.COLOR_RGB2BGR)
                    bgr_img = cv2.imread(imag)
                    img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
                #img = cv2.cvtColor(cv2.imread(imag), cv2.COLOR_BGR2RGB)
                #img = get_roi1(img)
                
                    mtcnn_detector = MTCNNFaceDetector()
                    bbox = mtcnn_detector.detect(img)

                    res = list(enumerate(bbox))
                # For each detected face, find the landmark.
                    if res != []:
                        img = roi_seg(img)
                        roi = mask_roi(img,bbox)
                    else :
                        print(imag)

"""
