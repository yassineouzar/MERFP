import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import csv

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statistics

def slidingAvg(N, stream):
    window = []
    list1 = []
    mean = np.mean(stream)
    # win = mean
    std_dev = np.std(stream)
    std = std_dev if (std_dev < 0.2 * mean) else 0.2 * mean
    # print(std)
    median = statistics.median(stream)
    # mc = most_common(stream)

    # for the first N inputs, sum and count to get the average     max(set(lst), key=lst.count)
    for i in range(N):
        window.append(stream[i])
    val = [mean if (abs(window[0] - mean) > std_dev) else window[0]]
    list1 = [mean if (abs(window[0] - mean) > std_dev) else window[0]]

    # afterwards, when a new input arrives, change the average by adding (i-j)/N, where j is the oldest number in the window
    for i in range(N, len(stream)):
        oldest = window[0]
        window = window[1:]
        window.append(stream[i])
        # window.append(window if (abs(stream[i] - stream[i] > std_dev)) else stream[i] )
        newest = window[0]
        val1 = oldest if (abs(newest - oldest) > std_dev) else newest
        val.append(val1)

    for i in range(1, len(val)):
        val[i] = list1[i - 1] if (abs(val[i] - val[i - 1]) > std) else val[i]
        list1.append(val[i])

        x = np.arange(0, len(stream))
    y = list1
    return y

def get_im1(path_im):
    global data
    list_dir = os.listdir(path_im)
    count = 0
    file_count = 0
    data = []
    global image
    image = []
    for i in range(int(len(list_dir))):
        list_dir1 = os.listdir(path_im + '/' + list_dir[i])
        Heart_rate_dir1 = []
        for j in range(int(len(list_dir1))):
            path_to_files = path_im + '/' + list_dir[i] + '/' + list_dir1[j]
            list_dir2 = os.listdir(path_to_files)
            for k in range(int(len(list_dir2))):
                path_to_files1 = path_im + '/' + list_dir[i] + '/' + list_dir1[j] + '/' + list_dir2[k]
                list_dir3 = os.listdir(path_to_files1)
                data.append(list_dir3)
    return data




def data_resampling(path_gt):
    path_im = '/media/ouzar1/Seagate Backup Plus Drive/Unzip_images'
    get_im1(path_im)
    df = pd.read_csv(path_gt, index_col=0)
    li = []
    global Heart_Rate
    for row in df[df.columns]:
        # d = df[df[row].notna()]
        x = df.loc[df[row].notna()]
        m = df[row].dropna(how='any')
        li.append(m)
    image = data
    HR = li
    #print(len(image), len(HR))
    A = []
    B = []
    Heart_Rate = []
    file_count = 0
    stream_hr = []
    batches = []
    batch = []
    for i in range(len(HR)):
        A = image[i]
        B = HR[i]
        #print(len(A), len(B))
        xx = len(B) - len(A)
        B.drop(B.tail(xx).index, inplace=True)

        heart_rate = B.tolist()
        print(type(A),type(heart_rate))
        slidingAvg(1, heart_rate)
        # file_count+=1
        # columns = ['HR '+ str(file_count)]
        # Heart_Rate.append(y)
        #for k in range((len(y)) // clip_size):
            # print(k)
            #batch = y[k * clip_size:(k + 1) * clip_size]
            #batches.append(batch)
        #print(len(batches))

        for j in y:
            Heart_Rate.append(j)
            hr = Heart_Rate
    label = []
    for h in hr:
        b = Counter(h)
        label.append(b.most_common()[0][0])
    print(len(label))
    label = np.array([i for i in label])
    print(hr)

    #print(len(batches))
        # print(len(stream_hr))
        # path_h = 'E:/HR/architecture/heart.csv'
        # with open(Heart_rate_resampled,'r') as file:
        # heart=[line.rstrip('\n') for line in file]
        # path_csv = os.path.join(os.path.splitext(Heart_rate_resampled)[0]  + ".csv")
        # with open(path_h,"w") as out_csv:
        # writer = csv.writer(out_csv, delimiter='\n')
        # writer.writerow(columns)
        # for word in Heart_Rate:
        # writer.writerow([word])
        # writer.writerow(Heart_Rate)
    return hr

def append_to_list_hr1(path_hr):
    file_count = 0
    list_dir = os.listdir(path_hr)
    for i in range(int(len(list_dir))):

        list_dir1 = os.listdir(path_hr + '/' + list_dir[i])
        Heart_rate_dir1 = []
        for j in range(int(len(list_dir1))):
            path_to_files = path_hr + '/' + list_dir[i] + '/' + list_dir1[j]
            list_dir2 = os.listdir(path_to_files)
            heart_rate = [filename for filename in list_dir2 if filename.startswith("Pulse")]
            file_count += 1
            columns = ['HR ' + str(file_count)]
            for hr in heart_rate:
                Heart_rate_resampled = os.path.join(path_hr + '/' + list_dir[i] + '/' + list_dir1[j] + '/' + hr)

                with open(Heart_rate_resampled, 'r') as file:
                    heart = [line.rstrip('\n') for line in file]
                path_csv = os.path.join(os.path.splitext(Heart_rate_resampled)[0] + ".csv")
                with open(path_csv, "w") as out_csv:
                    writer = csv.writer(out_csv, delimiter='\n')
                    writer.writerow(columns)
                    writer.writerow(heart)


def concat_list_hr(path_hr):
    list_dir = os.listdir(path_hr)
    global file_count
    global count
    file_count = 0
    data = {}
    global HR
    HR = []
    for i in range(int(len(list_dir))):

        list_dir1 = os.listdir(path_hr + '/' + list_dir[i])
        Heart_rate_dir1 = []
        for j in range(int(len(list_dir1))):
            file_count += 1

            path_to_files = path_hr + '/' + list_dir[i] + '/' + list_dir1[j]
            list_dir2 = os.listdir(path_to_files)
            heart_rate = [filename for filename in list_dir2 if filename.endswith("csv")]
            print(path_to_files)
            for hr in heart_rate:
                path_to_current_file = os.path.join(path_to_files, hr)
                HR.append(path_to_current_file)

                # df=[pd.read_csv(os.path.join(path_to_files, f)) for f in heart_rate]
                li = []
                columns = ['HR ' + str(file_count)]
                print(columns)
                for filename in HR:
                    # df = pd.read_csv(filename, sep='\t')
                    df = pd.read_csv(filename, index_col=None, parse_dates=True, infer_datetime_format=True)

                    li.append(df)

                    frame = pd.concat(li, axis=1, ignore_index=False)
                    # d=frame.dropna(how='any')
                # print(frame)
                frame.to_csv(r'/media/ouzar1/Seagate Backup Plus Drive/Heart_rate_csv.csv')


def append_to_list_hr(path_hr):
    file_count = 0
    data = []
    data1 = []
    Heart_Rate = []
    batches55 = []
    batches555 = []

    list_dir = sorted(os.listdir(path_hr))
    for i in range(int(len(list_dir))):

        list_dir1 = sorted(os.listdir(path_hr + '/' + list_dir[i]))
        Heart_rate_dir1 = []
        for j in range(int(len(list_dir1))):
            path_to_files = path_hr + '/' + list_dir[i] + '/' + list_dir1[j]
            list_dir2 = sorted(os.listdir(path_to_files))
            heart_rate = [filename for filename in list_dir2 if filename.startswith("Pulse Rate_BPM.t")]
            for hr in heart_rate:
                Heart_rate_resampled = os.path.join(path_hr + '/' + list_dir[i] + '/' + list_dir1[j] + '/' + hr)

                with open(Heart_rate_resampled, 'r') as file:
                    heart = [line.rstrip('\n') for line in file]
                    heart1 = [round(float(i)) for i in heart]
                    sps = round(1000 / 25)
                    resample = heart1[0::sps]

                    for n in resample:
                        data.append(round(n))

    for x in range((len(data)) // 50):
        batches_hr1 = data[x * 50:(x + 1) * 50]
        batches55.append(batches_hr1)
            # print(len(batches))
    for batch_hr in batches55:
                # b = Counter(batch_hr)
                # label = np.array(Counter(batches55).most_common()[0][0]).reshape(-1)
                #label1 = Counter(batch_hr).most_common()[0][0]
        label1 = np.mean(batch_hr)
        batches555.append(label1)
        # An "interface" to matplotlib.axes.Axes.hist() method
    n, bins, patches = plt.hist(x=batches555, bins='auto', alpha=0.9, rwidth=0.5)
    plt.grid(axis='y', alpha=1)
    plt.title('Histogram')
    plt.xlabel('Pulse Rate (bpm)')
    plt.ylabel('Number of Samples')

    maxfreq = n.max()
    # Set a clean upper y-axis limit.
    plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
    plt.show()
    # print(t)
    # plt.plot(t,a)

path_gt = '/home/ouzar1/Desktop/Dataset1/v4v_bp'

append_to_list_hr(path_gt)
#concat_list_hr(path_gt)
"""
                    for x in heart1:
                        data.append(round(x))
    for k in range((len(data)) // 25):
        data1.append(data[k * 25:(k + 1) * 25])

        # An "interface" to matplotlib.axes.Axes.hist() method
    n, bins, patches = plt.hist(x=data1, bins='auto', alpha=1, rwidth=10)
    plt.grid(axis='y', alpha=1)
    plt.title('Histogram')
    plt.xlabel('Frequency')
    maxfreq = n.max()
    # Set a clean upper y-axis limit.
    plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
    plt.show()
    # print(t)
    # plt.plot(t,a)
"""


"""
            for k in range((len(data)) // 50):
                batches_hr = data[k * 50:(k + 1) * 50]
                for l in batches_hr:
                    Heart_Rate.append(l)

    for x in range((len(Heart_Rate)) // 50):
        batches_hr1 = Heart_Rate[x * 50:(x + 1) * 50]
        batches55.append(batches_hr1)
            # print(len(batches))
    for batch_hr in batches55:
                # b = Counter(batch_hr)
                # label = np.array(Counter(batches55).most_common()[0][0]).reshape(-1)
                #label1 = Counter(batch_hr).most_common()[0][0]
        label1 = np.mean(batch_hr)
        batches555.append(label1)
        # An "interface" to matplotlib.axes.Axes.hist() method
    n, bins, patches = plt.hist(x=batches555, bins='auto', alpha=0.9, rwidth=0.5)
    plt.grid(axis='y', alpha=1)
    plt.title('Histogram')
    plt.xlabel('Pulse Rate (bpm)')
    plt.ylabel('Number of Samples')

    maxfreq = n.max()
    # Set a clean upper y-axis limit.
    plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
    plt.show()
    # print(t)
    # plt.plot(t,a)
"""
