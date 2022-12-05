"""
TOUCH PROSTHETICS - EMG CLASSIFICATION SYSTEM
    Author: Daniel Ingham
    Last Revision: 2022/12/02
    See Mindrove API documentation: https://docs.mindrove.com/UserAPI.html
"""

import csv
import os
import random
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from mindrove.board_shim import BoardShim, MindRoveInputParams, BoardIds
from mindrove.data_filter import DataFilter, FilterTypes, DetrendOperations
from tensorflow import keras

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def startup_print(startup: bool):
    """Prints the startup and menu options

    :param startup: prints the startup banner if true, else just prints the menu options
    :type startup: bool
    :return None
    """
    print(''.center(80, '-'))
    if startup:
        print("/‾‾‾‾‾/ /‾‾‾‾‾/ /‾/ /‾/ /‾‾_‾/ /‾/_/‾/".center(80, ' '))
        print("‾/ /‾  / /‾/ / / /_/ / / /    / __  /".center(80, ' '))
        print("/_/   /_____/ /_____/ /___‾/ /_/ /_/ ".center(80, ' '))
        print()
    print("OPTIONS:")
    print("1) Train classifier")
    print("2) Test classifier")
    print("3) Exit")


"""
======================================= EMG DATA ACQUISITION =======================================
"""


def timer_countdown(countdown_time_seconds: int, message: str) -> None:
    """Counts down in seconds from a specified number and prints the output to the terminal for each second

    :param countdown_time_seconds: time form which the countdown timer should start from
    :type countdown_time_seconds: int
    :param message: message to be printed on the last count (indicates the end of the countdown)
    :type message: str
    :return None
    """
    for seconds in range(countdown_time_seconds, -1, -1):
        start_time = time.perf_counter()
        end_time = time.perf_counter()
        while (end_time - start_time) < 1:
            end_time = time.perf_counter()
        if seconds == 0:
            print(message)
        else:
            print(seconds)


def print_movement(sample_num: int) -> None:
    """ Prints the the next movement to be performed when emg acquisition if taking place

    :param sample_num: the number of samples that have taken
    :type sample_num: int
    :return: None
    """
    if sample_num < 2000:
        print(int((2000 - sample_num) / 200), end=" ")
    elif sample_num < 5000:
        if sample_num == 2000:
            print("Rest")
    elif sample_num < 7000:
        print(int((7000 - sample_num) / 200), end=" ")
    elif sample_num < 10000:
        if sample_num == 7000:
            print("Flexion")
    elif sample_num < 12000:
        print(int((12000 - sample_num) / 200), end=" ")
    elif sample_num < 15000:
        if sample_num == 12000:
            print("Extension")
    elif sample_num == 15000:
        print("DONE")


def create_board() -> BoardShim:
    """ Creates an instance of the Mindrove board class for the Wifi armband using the parameters
        found from on https://docs.mindrove.com/SupportedBoards.html#mindrove-arb-arc

    :return: None
    """
    board_parameters = MindRoveInputParams()
    board_parameters.ip_port = 4210
    board_parameters.timeout = 10
    board_id = BoardIds.MINDROVE_WIFI_BOARD

    board = BoardShim(board_id, board_parameters)

    return board


def emg_data_acquisition() -> None:
    """ Reads data from the Mindrove armband an saves it into a csv file in the folder "emg_data/" to be used by the
        feature extraction function

    :return: None
    """
    while True:
        try:
            in_val = int(input("->How many rounds of EMG data collection? "))
            break
        except ValueError:
            print("Please input an INTEGER number")

    for x in range(in_val):
        # Start up Message
        message = 'ROUND ' + str(x + 1)
        print(message.center(80, "-"))

        # Create a unique file to store the EMG data
        file_name = "emg_data/EMG_" + str(time.time()) + ".csv"
        file = open(file_name, 'w', newline='')
        file.close()

        # Create Mindrove board object for the WiFi board for data acquisition
        board = create_board()
        board.disable_board_logger()
        emg_channels = board.get_emg_channels(BoardIds.MINDROVE_WIFI_BOARD)

        timer_countdown(3, "EMG READING INITIATED")

        # Prepare board for data logging session (NOTE: The armband starts saving data from this point on ward)
        board.prepare_session()
        while not board.is_prepared():
            pass

        # Initialise sample counter
        sample = 0

        # Start data stream
        board.start_stream()

        # Sample for 30 seconds: Rest > Flexion > Extension
        while sample < 15000:
            data_on_board = board.get_board_data_count()
            while data_on_board < 200:
                data_on_board = board.get_board_data_count()

            data = board.get_board_data(200)[0:8]

            sample += 200

            for channel in emg_channels:
                DataFilter.detrend(data[channel], DetrendOperations.CONSTANT.value)
                DataFilter.perform_bandpass(data[channel], BoardShim.get_sampling_rate(BoardIds.MINDROVE_WIFI_BOARD),
                                            51.0, 400.0, 5, FilterTypes.BUTTERWORTH.value, 0)
                DataFilter.perform_highpass(data[channel], BoardShim.get_sampling_rate(BoardIds.MINDROVE_WIFI_BOARD),
                                            0.5, 1, FilterTypes.BUTTERWORTH.value, 0)

            file = open(file_name, 'a', newline='')
            writer = csv.writer(file)
            writer.writerows(data.transpose())

            print_movement(sample)

        board.stop_stream()
        board.release_session()


"""
======================================= FEATURE EXTRACTION =======================================
"""
csv_heading = ["mav-ch1", "mav-ch2", "mav-ch3", "mav-ch4", "mav-ch5", "mav-ch6", "mav-ch7", "mav-ch8",
               "wl-ch1", "wl-ch2", "wl-ch3", "wl-ch4", "wl-ch5", "wl-ch6", "wl-ch7", "wl-ch8",
               "zc-ch1", "zc-ch2", "zc-ch3", "zc-ch4", "zc-ch5", "zc-ch6", "zc-ch7", "zc-ch8",
               "ssc-ch1", "ssc-ch2", "ssc-ch3", "ssc-ch4", "ssc-ch5", "ssc-ch6", "ssc-ch7", "ssc-ch8",
               "Movement"]


def read_csv(file_name: str, start_sample: int, end_sample: int, channels: list) -> None:
    """ Reads data from a provide csv file a stores the values in a 2D list to be used by the classifier algorithm

    :param file_name: name of the csv file to be read from
    :param start_sample: the starting sample point from which data acquisition should occur
    :param end_sample: the last sample point from which data acquisition should end
    :param channels: the list containing a list for each emg channel, to which the data will be extracted to
    :return: None
    """
    num_channels = len(channels)
    file = open(file_name)
    csv_reader = csv.reader(file)

    for i in range(end_sample):
        row = csv_reader.__next__()
        if i >= start_sample:
            for j in range(num_channels):
                channels[j].append(float(row[j]))

    file.close()


def get_list_average(input_list: list) -> float:
    """ Finds the average of all values in a list

    :param input_list: the from which the average should be found
    :return: float
    """
    total = 0
    for items in input_list:
        total += items

    return total / len(input_list)


def normalisation(sample_range: int, channels: list) -> None:
    """ Normalises the channel values relative to one another.

    :param sample_range: the range of samples from which the data in the input channels need to be normalised
    :param channels: 2D list containing the emg data for each 8 channels of the Mindrove armband
    :return: None
    """
    number_of_channels = len(channels)
    for i in range(number_of_channels):
        min_val = min(channels[i])
        max_val = max(channels[i])
        mean = get_list_average(channels[i])
        for j in range(sample_range):
            channels[i][j] = (channels[i][j] - mean) / (max_val - min_val)


def f(x: float, y: float, offset: float) -> int:
    """ Function used during feature extraction, specifically in the calculation of "Zero Crossing (ZC)" and
        "Slope Sign Change (SSC)". See "Algorithm of Myoelectric Signals Processing for the Control of Prosthetic
        Robotic Hands" in Journal of Computer Science & Technology, Volume 18.

    :param x: emg value for a single given sample
    :param y: emg value for the next sample following x
    :param offset: value used if the emg data has a constant offset.
    :return: int
    """
    xy = x * y
    if xy > offset:
        return 1
    else:
        return 0


def feature_extraction(sample_window: float, sample_range: int, channels: list) -> list:
    """ Extracts emg features from the raw emg data. See "Algorithm of Myoelectric Signals Processing for the Control
        of Prosthetic Robotic Hands" in Journal of Computer Science & Technology, Volume 18.

    :param sample_window: period of time from which a sample is taken (0.4 seconds)
    :param sample_range: range of sample values from the overall samples in the csv file
    :param channels: 2D list containing the emg data for each 8 channels of the Mindrove armband
    :return: list
    """

    num_channels = len(channels)

    mav = []    # Mean Absolute Value (MAV): the average value of the values sampled in the the sample window
    wl = []     # Waveform Length (WL): provides information on the waves amplitude, frequency and duration
    zc = []     # Zero Crossing (ZC): count the number of crosses by zero in the segment
    ssc = []    # Slope Sign Change (SSC): similar to the ZC feature but applied on the slope of the waveform

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

        mav.append(sample_window * mav_sum)
        wl.append(wl_sum)
        zc.append(zc_sum)
        ssc.append(ssc_sum)

    # Normalise
    normalised_mav = [elements / max(mav) for elements in mav]
    normalised_wl = [elements / max(wl) for elements in wl]
    normalised_zc = [elements / max(zc) for elements in zc]
    normalised_ssc = [elements / max(ssc) for elements in ssc]

    features = normalised_mav + normalised_wl + normalised_zc + normalised_ssc

    return features


def get_intervals(start: float, end: float, interval: float) -> list:
    """ Divides the range between the start and end values into equal parts and saves each increment between the start
        and the end in a list

    :param start: starting value
    :param end: ending values
    :param interval: size of intervals between the start and end value
    :return: list
    """
    sample_range = end - start
    divisions = int(sample_range / interval)
    intervals = []
    for x in range(divisions):
        intervals.append(start + x * interval)

    return intervals


def movement_encoding(label: str) -> int:
    """ Encodes movements into values using label encoding

    :param label: data label for a given data set
    :return: int
    """
    if label == "Rest":
        return 0
    if label == "Extension":
        return 1
    if label == "Flexion":
        return 2

    return None


def movement_decode(encoded_label: int) -> str:
    """ Decodes encoded data labels for better readability when printing

    :param encoded_label: encoded data label for a given data set
    :return:
    """
    if encoded_label == 0:
        return "Rest"
    if encoded_label == 1:
        return "Extension"
    if encoded_label == 2:
        return "Flexion"

    return None


def perform_feature_extraction() -> None:
    """ Using emg data stored in csv files, extracts emg features, and saves features into new emg files to be used
        by the classification algorithm for training and testing

    :return: None
    """
    channel = [[], [], [], [], [], [], [], []]
    sample_window = 0.4
    sample_rate = 500
    emg_data = {
        "Rest": [5, 9],
        "Flexion": [15, 19],
        "Extension": [25, 29]
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

    for files in res:
        file_name = 'emg_data/' + files
        print(files.center(80, "-"))

        for items in channel:
            items.clear()

        for key, value in emg_data.items():
            print(key)
            start_sample = int(value[0] * sample_rate)
            end_sample = int(value[1] * sample_rate)
            print(start_sample, end_sample)

            sample_intervals = get_intervals(value[0], value[1], sample_window)

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
    file = open('feature_extraction/training.csv', 'w', newline='')
    writer = csv.writer(file)
    writer.writerow(csv_heading)
    for items in training_data:
        writer.writerow(items)
    file.close()

    random.shuffle(testing_data)
    file = open('feature_extraction/testing.csv', 'w', newline='')
    writer = csv.writer(file)
    writer.writerow(csv_heading)
    for items in testing_data:
        writer.writerow(items)
    file.close()


"""
======================================= CLASSIFICATION ALGORITHM =======================================
"""


def get_dataset(filename: str) -> tuple:
    """ Reads data from a csv file and saves it into a pandas dataframe, which is then converted to a numpy array for
        use by the tensorflow neural network classifier

    :param filename: file name of the csv from which the data will be extracted from
    :return: tuple
    """
    emg_data = pd.read_csv(filename)
    emg_labels = emg_data.pop("Movement")

    emg_data = np.array(emg_data)
    emg_labels = np.array(emg_labels)

    return emg_data, emg_labels


def create_model() -> keras.models.Sequential:
    """ Creates the tensor flow neural network structure to be used to classify the emg data. Note that there are four
        features for each of the eight channels of the Mindrove armband, therefore the input layer should always have
        32 nodes. The number of nodes in the hidden layer can be adjusted according to performance requirements. The
        number of nodes in the output layer needs to correspond to the number of movements being classified

    :return: keras.models.Sequential
    """
    model = keras.models.Sequential([
        keras.layers.Dense(32, input_shape=(32,)),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(3),
    ])

    # loss and optimiser
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)  # soft max
    optimizer = keras.optimizers.Adam(learning_rate=0.01)  # Learning Rate
    metrics = ["accuracy"]

    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    return model


def train_classifier(x_train: np.array, y_train: np.array, x_test: np.array, y_test: np.array) -> None:
    """ Trains and tests the TensorFlow neural network using data training extracted from the emg armband

    :param x_train: emg feature data used in the training process
    :param y_train: labels for the emg feature data used in the training process
    :param x_test: emg feature data used in the testing process
    :param y_test: labels for the emg feature data used in the testing process
    :return: None
    """
    # Build the model
    classifier_model = create_model()

    batch_size = 10
    epochs = 10

    # Start the training
    print("TRAINING".center(80, "-"))
    classifier_model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=True, verbose=2)

    # Save the model
    classifier_model.get_weights()
    classifier_model.save_weights('classification_model/emg_classifier_weights.h5')

    # Evaluate Model
    print("EVALUATING MODEL".center(80, "-"))
    classifier_model.evaluate(x_test, y_test, verbose=2)

    predictions = classifier_model(x_test)
    predictions = tf.nn.softmax(predictions)
    first_twenty_predictions = predictions[0:20]
    print("Predictions\t", np.argmax(first_twenty_predictions, axis=1))
    print("Answers\t", y_test[0:20])


"""
======================================= TEST CLASSIFICATION ALGORITHM =======================================
"""


def test_classifier(test_duration_seconds: int) -> None:
    """ Tests the TensorFlow neural network classifier on live data and prints the classifications to the terminal

    :param test_duration_seconds: duration for which testing should occur
    :return: None
    """
    new_model = create_model()

    new_model.load_weights("classification_model/emg_classifier_weights.h5")

    timer_countdown(3, "EMG READING INITIATED")

    # Prepare board for data logging session (NOTE: The armband starts saving data from this point on ward)
    board = create_board()
    board.disable_board_logger()  # Removes annoying Mindrove Json object terminal logging :)
    emg_channels = board.get_emg_channels(BoardIds.MINDROVE_WIFI_BOARD)

    board.prepare_session()
    while not board.is_prepared():
        pass

    # Start data stream
    board.start_stream()
    t_start = time.perf_counter()
    t_end = time.perf_counter()

    while (t_end - t_start) < test_duration_seconds:
        data_on_board = board.get_board_data_count()
        while data_on_board < 200:
            data_on_board = board.get_board_data_count()

        data = board.get_board_data(200)[0:8]

        for channel in emg_channels:
            DataFilter.detrend(data[channel], DetrendOperations.CONSTANT.value)
            DataFilter.perform_bandpass(data[channel], BoardShim.get_sampling_rate(BoardIds.MINDROVE_WIFI_BOARD),
                                        51.0, 400.0, 5, FilterTypes.BUTTERWORTH.value, 0)
            DataFilter.perform_highpass(data[channel], BoardShim.get_sampling_rate(BoardIds.MINDROVE_WIFI_BOARD),
                                        0.5, 1, FilterTypes.BUTTERWORTH.value, 0)

        features = feature_extraction(0.4, 200, data[0:8])
        np_features = np.array([features])
        val = np.argmax(new_model(np_features), axis=1)
        str_val = movement_decode(val)
        print(str_val)

        t_end = time.perf_counter()

    board.stop_stream()
    board.release_session()


"""
======================================= MAIN FUNCTION =======================================
"""

if __name__ == "__main__":
    startup_print(True)
    while True:
        print(''.center(80, '-'))

        user_in = input("Select an Option: ")

        if user_in == "1":
            print('CLASSIFIER TRAINING'.center(80, ' '))
            emg_data_acquisition()
            perform_feature_extraction()

            data_train, label_train = get_dataset(
                "feature_extraction/training.csv")
            data_test, label_test = get_dataset(
                "feature_extraction/testing.csv")

            print("Input data shape", data_train.shape, label_train.shape)
            train_classifier(data_train, label_train, data_test, label_test)
        elif user_in == "2":
            print('CLASSIFIER TESTING'.center(80, ' '))
            test_classifier(30)
        else:
            quit()
        time.sleep(5)
        startup_print(False)
