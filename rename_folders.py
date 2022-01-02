import os
import numpy as np
import csv
from shutil import copyfile

def Exp_segment(path_ppg):

    list_dir = sorted(os.listdir(path_ppg))
    for i in range(int(len(list_dir))):
        path_to_AU = os.path.join(path_ppg,list_dir[i])

        f_name, f_ext = os.path.splitext(path_to_AU)
        subjects = os.path.basename(os.path.normpath(path_to_AU))
        # new_name = os.path.join(os.path.dirname(path_to_AU), subjects.replace(subjects[0], 'AUG-'))
        new_name = os.path.join(os.path.dirname(path_to_AU), "AUG-"+subjects)

        os.rename(path_to_AU, new_name)
        print(new_name)




path_ppg = '/home/ouzar1/Documents/pythonProject/BP4D-Expressive-Segment/multimodal/New folder'

Exp_segment(path_ppg)


"""
                if (os.path.isfile(ES_src)):

                    copyfile(ES_src, ES_dst)
                    print(ES_src, ES_dst)

                else:
                    print(ES_src + " not found")
"""