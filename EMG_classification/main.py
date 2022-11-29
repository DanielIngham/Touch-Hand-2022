"""
Time intervals from emg csv data:
- Rest:                  4 - 10 seconds
- Extension:            16 - 20 seconds
- Flexion:              24 - 30 seconds
- Ulnar Deviation:      34 - 40 seconds
- Radial Deviation:     44 - 50 seconds
- Grip:                 54 - 60 seconds
- Abduction of Fingers: 64 - 70 seconds
- Adduction of Fingers: 74 - 80 seconds
- Supination:           84 - 90 seconds
- Pronation             94 - 100 seconds
"""

import csv
import matplotlib.pyplot as plt
import random
import os

"""
Helper Functions
"""
csv_heading = ["mav-ch1","mav-ch2","mav-ch3","mav-ch4",
               "wl-ch1","wl-ch2","wl-ch3","wl-ch4",
               "zc-ch1","zc-ch2","zc-ch3","zc-ch4",
               "ssc-ch1","ssc-ch2","ssc-ch3","ssc-ch4",
               "Movement"]

def read_csv(file_name, start_sample, end_sample, channels):
    num_channels = len(channels)
    file = open(file_name)
    csv_reader = csv.reader(file)

    for i in range(end_sample):
        row = csv_reader.__next__()
        if i >= start_sample:
            for j in range(num_channels):
                channels[j].append(float(row[j]))

    file.close()


def avg(val_list):
    total = 0
    for items in val_list:
        total += items

    return total / len(val_list)


""" Normalisation """


def normalisation(sample_range, channels):
    num_channels = len(channels)
    for i in range(num_channels):
        min_val = min(channels[i])
        max_val = max(channels[i])
        mean = avg(channels[i])
        for j in range(sample_range):
            channels[i][j] = (channels[i][j] - mean) / (max_val - min_val)


def normalise_features():
    pass


def f(x, y, umbral) -> int:
    xy = x * y
    if xy > umbral:
        return 1
    else:
        return 0


""" 
Feature Extraction 
-   Feature extraction derived from the expressions found in 
    "Signal Processing for Robotic Hand" by Rodrigo E. Russo
"""


def feature_extraction(sample_window, sample_range, channels):
    n = sample_window
    num_channels = len(channels)

    mav = []
    wl = []
    zc = []
    ssc = []

    for i in range(num_channels):
        mav_sum = 0
        wl_sum = 0
        zc_sum = 0
        ssc_sum = 0

        for j in range(sample_range):
            mav_sum += abs(channels[i][j])
            if j < (sample_range - 1):
                wl_sum += abs(channels[i][j + 1] - channels[i][j])
                zc_sum += f(channels[i][j + 1], channels[i][j], 0)
            if j < (sample_range - 2):
                ssc_sum += f(channels[i][j + 1] - channels[i][j], channels[i][j + 2] - channels[i][j + 1], 0)

        mav.append(n * mav_sum)
        wl.append(wl_sum)
        zc.append(zc_sum)
        ssc.append(ssc_sum)

    # Normalise
    n_mav = [x/max(mav) for x in mav]
    n_wl = [x/max(wl) for x in wl]
    n_zc = [x/max(zc) for x in zc]
    n_ssc = [x/max(ssc) for x in ssc]
    features = n_mav + n_wl + n_zc + n_ssc

    # features = mav + wl + zc + ssc
    return features


def plot_emg(channels, title):
    plt.figure()
    plt.title(title)
    num_channels = len(channels)

    for k in range(num_channels):
        x_values = channels[k]
        plt_label = "Channel " + str(k)
        plt.plot(range(len(x_values)), x_values, label=plt_label)

    plt.legend()
    plt.show()


def get_intervals(start, end, interval):
    sample_range = end - start
    divisions = int(sample_range / interval)
    intervals = []
    for x in range(divisions):
        intervals.append(start + x * interval)
    return intervals

def movement_encoding(label: str):
    if label == "Rest":
        return 0
    if label == "Extension":
        return 1
    if label == "Flexion":
        return 2
    return None

def main():
    channel = [[], [], [], []]
    sample_window = 0.1
    sample_rate = 2000
    emg_data = {
        "Rest": [5, 9],
        "Extension": [15, 19],
        "Flexion": [25, 29]
    }

    # EMG csv data path
    dir_path = 'emg_data/'

    # list to store files
    res = []

    # Iterate directory
    for path in os.listdir(dir_path):
        # check if current path is a file
        if os.path.isfile(os.path.join(dir_path, path)):
            res.append(path)
    print(res)

    training_data = []
    testing_data = []

    res.clear()
    res.append('10_filtered.csv')

    for y in range(5):
        file_name = 'emg_data/' + '10_filtered.csv'
        print('10_filtered.csv'.center(80, "-"))

        read_csv(file_name, (4+134*y)*2000, (100+134*y)*2000, channel)
        plot_emg(channel, "EMG")

        for items in channel:
            items.clear()

        for key, value in emg_data.items():
            print(key)

            sample_intervals = get_intervals(value[0]+134*y, value[1]+134*y, sample_window)

            for x in range(len(sample_intervals) - 1):
                start_sample = int(sample_intervals[x] * sample_rate)
                end_sample = int(sample_intervals[x + 1] * sample_rate)
                sample_range = end_sample - start_sample

                for items in channel:
                    items.clear()

                read_csv(file_name, start_sample, end_sample, channel)

                features = (feature_extraction(sample_window, sample_range, channel))
                features.append((movement_encoding(key)))

                if (x % 3) == 0:
                    testing_data.append(features)
                else:
                    training_data.append(features)

    random.shuffle(training_data)
    f = open('results/training.csv', 'w', newline='')
    writer = csv.writer(f)
    writer.writerow(csv_heading)
    for items in training_data:
        writer.writerow(items)
    f.close()

    random.shuffle(testing_data)
    f = open('results/testing.csv', 'w', newline='')
    writer = csv.writer(f)
    writer.writerow(csv_heading)
    for items in testing_data:
        writer.writerow(items)
    f.close()


if __name__ == "__main__":
    main()
