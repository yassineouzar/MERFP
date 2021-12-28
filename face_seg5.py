import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import math
import numpy as np
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
import cv2
from imutils import face_utils
import dlib
# from tensorflow.python.keras.utils import np_utils
# from keras import backend as K
# K.common.image_dim_ordering()
# K.set_image_dim_ordering('th')
from keras.layers import *
from keras.layers.advanced_activations import LeakyReLU
from keras.activations import relu
from keras.initializers import RandomNormal
from keras.applications import *
import keras.backend as K
#from face_detector import *
#from detection_viewer import DetectionViewer
#from image_cropper import ImageCropper
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

    # cv2.imshow('img', (mask[:, :, np.newaxis] * im.astype(np.float64)).astype(np.uint8))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.imwrite('D:/roi_a_verifier/1/T11/face_seg.jpg', (mask[:, :, np.newaxis] * im.astype(np.float64)).astype(np.uint8))
    image = (mask[:, :, np.newaxis] * im.astype(np.float64)).astype(np.uint8)
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # cv2.imshow('img', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # image1 = np.resize(image,(110,260,3))
    # print(image1.shape, image1.ndim)
    return image


# https://towardsdatascience.com/facial-mapping-landmarks-with-dlib-python-160abcf7d672
def draw_convex_hull(im, points, color):
    points = cv2.convexHull(points)
    cv2.fillConvexPoly(im, points, color=color)


# from get_roi import get_roi
def mask_roi(image, bbox):
    p = '/tmp/.x2go-ouzar1/media/disk/_cygdrive_C_Users_ouzar1_DOCUME1_SHAREF1/Face_segmentation/shape_predictor_68_face_landmarks.dat'
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(p)

    # bbox = dlib_detector.detect()
    # dlib_detector = DlibCVFaceDetector()
    # mtcnn_detector = MTCNNFaceDetector()

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
                shape = face_utils.shape_to_np(shape)
                # for (k,l) in shape:
                # cv2.circle(image, (k, l), 2, (0, 255, 25), 2)
                # cv2.circle(image, (95, 302), 2, (0, 255, 25), 2)
                # cv2.rectangle(image, (rect.left(), rect.top()), (rect.right(), rect.bottom()), (0, 255, 0), 3)

                # cv2.imwrite('D:/roi_a_verifier/1/T11/face_lm.jpg', image)
                pts = np.array(
                    [shape[31], shape[32], shape[33], shape[34], shape[35], shape[26], shape[25], shape[24], shape[23],
                     shape[22], shape[21], shape[20], shape[19], shape[18], shape[17]], np.int32)
                # image1 = cv2.polylines(image, [pts], 2, (255, 255, 255), 5)

                pts1 = np.array(
                    [shape[48], shape[49], shape[51], shape[52], shape[53], shape[54], shape[55], shape[56], shape[57],
                     shape[58], shape[59]])

                # draw_convex_hull(image, pts1, color=0)
                # draw_convex_hull(image, pts, color=0)
                # image = image[rect.top():rect.bottom(), rect.left():rect.right()]

                draw_convex_hull(image, pts1, color=0)
                draw_convex_hull(image, pts, color=0)

        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        #img_crop = ImageCropper(image)
        #im = img_crop.crop(bbox)

        #im = cv2.resize(im, (160, 240))

        # print(im.shape)
        return image
        # im = cv2.resize(im,(120,120))
        # print(im.shape)
        # return im


def get_roi1(image):
    detector = MTCNN()
    global image1

    result = detector.detect_faces(image)
    if result != []:
        for person in result:
            bounding_box = person['box']
            keypoints = person['keypoints']

            # detector.detect_faces(image)

            # result = detector.detect_faces(image)

            # Result is an array with all the bounding boxes detected. We know that for 'ivan.jpg' there is only one.
            # bounding_box = result[0]['box']

            # keypoints = result[0]['keypoints']

            cv2.circle(image, (keypoints['left_eye']), 2, (0, 155, 255), 2)
            cv2.circle(image, (keypoints['right_eye']), 2, (0, 155, 255), 2)
            cv2.circle(image, (keypoints['nose']), 2, (0, 155, 255), 2)
            cv2.circle(image, (keypoints['mouth_left']), 2, (0, 155, 255), 2)
            cv2.circle(image, (keypoints['mouth_right']), 2, (0, 155, 255), 2)

            image = image[bounding_box[1]: bounding_box[1] + bounding_box[3],
                    bounding_box[0]: bounding_box[0] + bounding_box[2]]

            left_eye = []
            left_eye = keypoints['left_eye']
            a1 = left_eye[0]
            a2 = left_eye[1]
            x1 = a1 - bounding_box[0]

            right_eye = []
            right_eye = keypoints['right_eye']
            b1 = right_eye[0]
            x2 = b1 - bounding_box[0]
            b2 = right_eye[1]

            y2 = int((a2 - bounding_box[1]) / 2)

            # cv2.rectangle(image, (x1, 20), (x2, y2) , (0,155,255), 2)

            # image = image[20 : y2 , x1 : x2]
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # image = img_to_array(image)
            # image = cv2.resize(image,(200,100))
            # print(image.shape)
            cv2.imshow('img', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            # image1 = np.resize(image,(110,260,3))
            # print(image1.shape, image1.ndim)
    return image


# from get_roi import get_roi


def get_im_train(path_im, path_save_im):
    global data_train
    list_dir = os.listdir(path_im)
    count = 0
    file_count = 0
    data_train = []
    train_data = []
    train_data1 = []

    global image1
    image1 = []
    for i in range(int(len(list_dir))):
    #for i in range(35,140):
        list_dir1 = os.listdir(path_im + '/' +  list_dir[i])
        print(list_dir1)

        list_dir_save1 = path_save_im + '/' +  list_dir[i]
        if not os.path.exists(list_dir_save1):
            os.makedirs(list_dir_save1)
        Heart_rate_dir1=[]
        for j in range(int(len(list_dir1))):
            path_to_files=path_im + '/' +  list_dir[i] + '/' + list_dir1[j]
            list_dir2 = os.listdir(path_to_files)
            print(list_dir2)

            for k in range(int(len(list_dir2))):
                path_to_files1 = path_im + '/' + list_dir[i] + '/' + list_dir1[j] + '/' + list_dir2[k]
                list_dir3 = os.listdir(path_to_files1)
                list_dir_save2 = path_save_im + '/' +  list_dir[i] + '/' + list_dir2[k]
                print(path_to_files1)

                if not os.path.exists(list_dir_save2):
                    os.makedirs(list_dir_save2)
                for im in sorted(list_dir3):

                    imag = os.path.join(path_to_files1, im)
                    imag1 = os.path.join(list_dir_save2, im)
                    img = cv2.cvtColor(cv2.imread(imag), cv2.COLOR_RGB2BGR)
                    img = roi_seg(img)
                    #y, x = img.shape[:2]
                    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

                    y, x = img.shape[:2]
                    # gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                    h = []
                    w = []
                    for m in range(x):
                        for l in range(y):

                            b, g, r = (img[l, m])
                            if ([b, g, r] >= [30, 30, 30]):
                                w.append(m)
                                h.append(l)
                                # mask = [b,g,r]>=[15,15,15]
                    x1, x2, y1, y2 = min(w), max(w), min(h), max(h)
                    img = img[y1:y2, x1:x2]

                    #print(x1,x2,y1,y2)

                    img = cv2.resize(img, (120, 160))
                    #cv2.imshow('img', img)
                    #cv2.waitKey(0)
                    #cv2.destroyAllWindows()
                    #print(img.shape)
                    cv2.imwrite(imag, img)
                    count += 1
                    print(count)

path_im = '/home/ouzar1/Desktop/signal_test'
path_save_im = '/home/ouzar1/Desktop/ROI'

print("begin1")
get_im_train(path_im, path_save_im)
print("finished")

"""
                    y, x = img.shape[:2]
                    # gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                    for i in range(x):
                        for j in range(y):
                            b, g, r = (img[j, i])
                            if ([b, g, r] >= [50, 50, 50]):
                                w.append(i)
                                h.append(j)
                                # mask = [b,g,r]>=[15,15,15]
                    x1, x2, y1, y2 = min(w), max(w), min(h), max(h)

                    img = img[y1:y2, x1:x2]

                    img = cv2.resize(img, (260, 348))
                    #print(img.shape)
                    #cv2.imshow('img', img)
                    #cv2.waitKey(0)
                    #cv2.destroyAllWindows()
                    cv2.imwrite(imag1, img)
                    count += 1
                    print(count)

"""
