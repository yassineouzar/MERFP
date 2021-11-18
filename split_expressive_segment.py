import os
import numpy as np
import csv
from shutil import copyfile

def Exp_segment(path_im, path_AU, path_ES):

    list_dir = sorted(os.listdir(path_AU))
    for i in range(int(len(list_dir))):
        path_to_AU = os.path.join(path_AU,list_dir[i])
        path_to_im = os.path.join(path_im, list_dir[i])
        path_to_ES = os.path.join(path_ES, list_dir[i])
        dir_im = os.path.splitext(path_to_im)[0].replace("_","/")
        dir_ES = os.path.splitext(path_to_ES)[0].replace("_","/")
        if not os.path.exists(dir_ES):
            os.makedirs(dir_ES)
        #print(dir_ES, dir_im)


        with open (path_to_AU) as csv_file:
            reader = csv.reader(csv_file)
            next(reader)  # skip header
            for row in reader:
                row = str(row[0]).zfill(4)
                ES_src = os.path.join(dir_im, row + '.jpg')
                ES_dst = os.path.join(dir_ES, row + '.jpg')

                if (os.path.isfile(ES_src)):

                    copyfile(ES_src, ES_dst)
                    print(ES_src, ES_dst)

                else:
                    print(ES_src + " not found")




path_im = '/media/bousefsa1/My Passport/BD PPG/2 bases publiques/BP4D-ROI-segmented'
path_AU = '//media/bousefsa1/Seagate Backup Plus Drive/BP4D+_v0.2/AUCoding/AU_OCC'
path_ES = '/media/bousefsa1/My Passport/BD PPG/2 bases publiques/BP4D-Expressive-Segment'

Exp_segment(path_im, path_AU, path_ES)


