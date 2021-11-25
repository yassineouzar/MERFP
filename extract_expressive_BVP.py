import os
import numpy as np
import csv
from shutil import copyfile

def Exp_segment(path_bp, path_AU, path_ES_bp):
    data = []
    list_dir = sorted(os.listdir(path_AU))
    for i in range(int(len(list_dir))):
        path_to_AU = os.path.join(path_AU,list_dir[i])
        path_to_bp = os.path.join(path_bp, list_dir[i])
        path_to_ES_bp = os.path.join(path_ES_bp, list_dir[i])
        dir_bp = os.path.splitext(path_to_bp)[0].replace("_","/")
        dir_ES_bp = os.path.splitext(path_to_ES_bp)[0].replace("_","/")
        if not os.path.exists(dir_ES_bp):
            os.makedirs(dir_ES_bp)

        print(dir_ES_bp)
        list_dir_bp = sorted(os.listdir(dir_bp))
        blood_pressure = [filename for filename in list_dir_bp if filename.startswith("BP_mmHg.t")]

        for bp in blood_pressure:
            bp_resampled = os.path.join(dir_bp + '/' + bp)
            bp_expressive = os.path.join(dir_ES_bp + '/' + bp)

            with open(bp_resampled, 'r') as file:
                bpi= [line.rstrip('\n') for line in file]
                #bpr = [round(float(i)) for i in bpi]
                bpr = [float(i) for i in bpi]
                sps = round(1000 / 25)
                resample = bpr[0::sps]
                #print(bp_resampled)
        range_bp = []
        with open(path_to_AU) as csv_file:
            reader = csv.reader(csv_file)
            next(reader)  # skip header
            for row in reader:
                row = row[0]
                range_bp.append(row)
        #print(range_bp[0], range_bp[-1])
        #print(resample[int(range_bp[0]) : int(range_bp[-1])])
        #print(len(resample)-int(range_bp[-1]))

        with open(bp_expressive, 'w') as f:
            for x in resample[int(range_bp[0]) : int(range_bp[-1])]:
                #print(x)
                f.write("%s \n" % x)





path_bp = '/home/ouzar1/Documents/pythonProject/Physiology'
path_AU = '/home/ouzar1/Documents/pythonProject/AUCoding/AU_OCC'
path_ES_bp = '/home/ouzar1/Documents/pythonProject/BP4D-Expressive-Segment-bp'

Exp_segment(path_bp, path_AU, path_ES_bp)


