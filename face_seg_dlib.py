import os
import numpy as np
import cv2
import dlib
from imutils import face_utils



# from get_roi import get_roi
def mask_roi(image):
    p = 'E:/BP4D/test/shape_predictor_81_face_landmarks.dat'
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(p)
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

                pts = np.array([shape[0], shape[1], shape[2], shape[3], shape[4], shape[5], shape[6], shape[7], shape[8],
                            shape[9], shape[10], shape[11], shape[12], shape[13], shape[14], shape[15], shape[16], shape[78],
                    shape[74], shape[79], shape[73], shape[72], shape[80], shape[71], shape[70], shape[69], shape[68],
                     shape[76], shape[75], shape[77]], np.int32)

                mask = np.zeros_like(image)
                mask = cv2.fillPoly(mask, [pts], (255, 255, 255))
                image = cv2.bitwise_and(image, mask)
        return image

def get_im_train(path_im, path_save_im):
    global data_train
    list_dir = sorted(os.listdir(path_im))

    count = 0
    file_count = 0
    data_train = []
    train_data = []
    train_data1 = []

    global image1
    image1 = []
    #for i in range(int(len(list_dir))):
    for i in range(175,180):
        list_dir1 = sorted(os.listdir(path_im + '/' +  list_dir[i]))
        list_dir_save1 = path_save_im + '/' +  list_dir[i]
        if not os.path.exists(list_dir_save1):
            os.makedirs(list_dir_save1)
        Heart_rate_dir1=[]
        for j in range(int(len(list_dir1))):
            path_to_files = path_im + '/' + list_dir[i] + '/' + list_dir1[j]
            list_dir2 = os.listdir(path_to_files)
            list_dir_save1 = path_save_im + '/' + list_dir[i] + '/' + list_dir1[j]
            if not os.path.exists(list_dir_save1):
                os.makedirs(list_dir_save1)
            for im in sorted(list_dir2):
                imag = os.path.join(path_to_files, im)
                imag1 = os.path.join(list_dir_save1, im)
                print(imag1)
                img = cv2.imread(imag)

                img = mask_roi(img)
                #cv2.imshow('img', img)
                #cv2.waitKey(0)
                #cv2.destroyAllWindows()
                y, x = img.shape[:2]
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
                #img = cv2.pyrUp(img)

                img = cv2.resize(img, (120, 160))
                #print(img.shape)

                #cv2.imwrite(imag1, img)
                count += 1
                print(count)
                #cv2.imshow('img', img)
                #cv2.waitKey(0)
                #cv2.destroyAllWindows()

path_im = 'E:/BP4D/manhob hci/resample'
path_save_im = 'G:/MAHNOB_seg'

print("begin1")
get_im_train(path_im, path_save_im)
print("finished")


"""
/home/ouzar1/Desktop/Dataset1/model/Ubfc_25fps/15/T0/0395.png
(148, 111, 3)
1
/home/ouzar1/Desktop/Dataset1/model/Ubfc_25fps/15/T0/1369.png
(141, 102, 3)
2
/home/ouzar1/Desktop/Dataset1/model/Ubfc_25fps/15/T0/0977.png
"""