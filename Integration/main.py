"""
TOUCH PROSTHETICS - EMG CLASSIFICATION SYSTEM
    Author: Daniel Ingham
    Last Revision: 2022/12/05
    See Mindrove API documentation for armband API information: https://docs.mindrove.com/UserAPI.html
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

# =========== USER VARIABLES ===========
# Add or remove desired movements that need to be classified (NOTE: DO NOT REMOVE "Rest")
movements = ["Rest", "Flexion", "Extension"]

# Set the batch size and number of epochs the program should use when training the neural network
batch_size = 10
epochs = 50

# Set the duration for which the testing of the classifier should go on for
testing_time_seconds = 30

# =======================================
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def startup_print(startup: bool) -> None:
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


def print_movement(number_of_samples: int) -> None:
    """ Prints the the next movement to be performed when emg acquisition if taking place

    :param number_of_samples: the number of samples that have taken
    :type number_of_samples: int
    :return: None
    """
    global movements
    index = 0
    if number_of_samples != 5000*len(movements):
        for _ in movements:
            index = int(np.floor((number_of_samples+1)/5000))
        sample_5000 = number_of_samples-5000*index

        if sample_5000 < 2000:
            print(int((2000 - sample_5000) / 200), end=" ")
        elif sample_5000 < 5000:
            if sample_5000 == 2000:
                print(movements[index])


def create_board() -> BoardShim:
    """ Creates an instance of the Mindrove board class for the Wifi armband using the parameters
        found from on https://docs.mindrove.com/SupportedBoards.html#mindrove-arb-arc

    :return: None
    """
    board_parameters = MindRoveInputParams()
    board_parameters.ip_port = 4210
    board_parameters.timeout = 10

    board = BoardShim(BoardIds.MINDROVE_WIFI_BOARD, board_parameters)

    return board


def emg_data_acquisition(total_movements: int) -> None:
    """ Reads data from the Mindrove armband an saves it into a csv file in the folder "emg_data/" to be used by the
        feature extraction function
    :param total_movements: the number of movements to be classified
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
        print_movement(sample)

        # Start data stream
        board.start_stream()

        # Sample each movement for 5000 samples: 2000 inactive, 3000 movement
        while sample < total_movements*5000:
            # Wait until Mindrove armband has 200 raw emg samples
            data_on_board = board.get_board_data_count()
            while data_on_board < 200:
                data_on_board = board.get_board_data_count()

            # Get 200 samples of only the emg channels (channels 0 - 7)
            data = board.get_board_data(num_samples=200)[0:8]

            # Increase the sample counter
            sample += 200

            # For each channel of the EMG armband, apply filters to the raw emg data
            for channel in emg_channels:
                # Detrend data: remove linear offsets caused by the surface contact electrodes
                DataFilter.detrend(data=data[channel], detrend_operation=DetrendOperations.CONSTANT.value)
                # 5th order Bandpass Filter: 50Hz - 400Hz, see “Digital filtering of emg signals”, V. Zschorlich
                DataFilter.perform_bandpass(data=data[channel],
                                            sampling_rate=BoardShim.get_sampling_rate(BoardIds.MINDROVE_WIFI_BOARD),
                                            center_freq=225.0, band_width=350.0, order=5,
                                            filter_type=FilterTypes.BUTTERWORTH.value, ripple=0)
                # 1st order Highpass Filter: 0.5Hz cut-off (DC blocker). Mindrove recommendation
                DataFilter.perform_highpass(data=data[channel],
                                            sampling_rate=BoardShim.get_sampling_rate(BoardIds.MINDROVE_WIFI_BOARD),
                                            cutoff=0.5, order=1, filter_type=FilterTypes.BUTTERWORTH.value, ripple=0)

            # Write emg data to disk
            file = open(file_name, 'a', newline='')
            writer = csv.writer(file)
            writer.writerows(data.transpose())

            # Update movement interface
            print_movement(sample)

        # End Mindrove armband streaming session
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
    number_of_channels = len(channels)
    file = open(file_name)
    csv_reader = csv.reader(file)

    for i in range(end_sample):
        row = csv_reader.__next__()
        if i >= start_sample:
            for j in range(number_of_channels):
                channels[j].append(float(row[j]))

    file.close()


def get_list_average(input_list: list) -> float:
    """ Finds the average of all values in a list

    :param input_list: the from which the average should be found
    :return: float, corresponding to the average value of the list
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
    :return: int, corresponding to the output of the function (either 0 or 1)
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
    :return: list, containing all the features extracted from the single sample window for all 8 channels of the
             Mindrove armband
    """

    number_of_channels = len(channels)

    mav = []    # Mean Absolute Value (MAV): the average value of the values sampled in the the sample window
    wl = []     # Waveform Length (WL): provides information on the waves amplitude, frequency and duration
    zc = []     # Zero Crossing (ZC): count the number of crosses by zero in the segment
    ssc = []    # Slope Sign Change (SSC): similar to the ZC feature but applied on the slope of the waveform

    for i in range(number_of_channels):
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

    # Normalisation of the features for input into the classifier algorithm
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
    :return: list, containing the each incremental value from "start" to "end". E.g: [start, start+interval, ..., end]
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
    :return: int, label encoded label for a the given movement dataset
    """
    global movements
    return movements.index(label)


def movement_decode(encoded_label: int) -> str:
    """ Decodes encoded data labels for better readability when printing

    :param encoded_label: encoded data label for a given data set
    :return: str, corresponding string to the label encoded values
    """
    global movements
    return movements[encoded_label]


def perform_feature_extraction() -> None:
    """ Using emg data stored in csv files, extracts emg features, and saves features into new emg files to be used
        by the classification algorithm for training and testing.
        The csv file of the raw emg data has the following layout:
        |     INACTIVE   |   MOVEMENT 1   |  ...
        | 2000 Samples   | 3000 Samples   |  ...

    :return: None
    """
    global movements
    global csv_heading
    # Board specific variables (Mindrove WiFi board)
    channel = [[], [], [], [], [], [], [], []]  # 8 emg channels on the Mindrove armband
    sample_window_time = 0.4
    sample_rate = 500
    sample_window_samples = sample_window_time * sample_rate

    # csv file path for emg data path
    directory_path = 'emg_data/'

    # List to store files
    folder = []

    # Iterate directory
    for path in os.listdir(directory_path):
        # Check if current path is a file
        if os.path.isfile(os.path.join(directory_path, path)):
            folder.append(path)

    # Lists that will split raw emg data into training and testing data set (66:33)
    training_data = []
    testing_data = []

    # Iterate each file in emg data directory
    for files in folder:
        # File names for reading
        file_name = directory_path + files
        print(files.center(80, "-"))

        # Clear list for each iteration
        for items in channel:
            items.clear()

        # For each specified movement in the global list "movements"
        for movement in movements:
            print(movement)

            # Find the starting point for the movement in the emg csv file (see csv layout above)
            index = movements.index(movement)
            start_sample = 2000 + index*5000
            end_sample = start_sample + 3000
            print(start_sample, end_sample)

            # Get list of sample intervals based of the start and end points specified
            sample_intervals = get_intervals(start_sample, end_sample, sample_window_samples)

            # For each interval
            for x in range(len(sample_intervals) - 1):
                # Extract sub-intervals
                start_sample = int(sample_intervals[x])
                end_sample = int(sample_intervals[x + 1])
                sample_range = end_sample - start_sample

                # Clear the channels of previous readings
                for items in channel:
                    items.clear()

                # Extract data from emg data file for each sample interval
                read_csv(file_name, start_sample, end_sample, channel)

                # Retrieve and save features from the data with corresponding label
                features = feature_extraction(sample_window_time, sample_range, channel)
                features.append((movement_encoding(movement)))

                # Split data 66:33 between training and testing data
                if (x % 3) == 0:
                    testing_data.append(features)
                else:
                    training_data.append(features)

    # Randomly shuffle the training data and save into csv file (for better training)
    random.shuffle(training_data)
    file = open('feature_extraction/training.csv', 'w', newline='')
    writer = csv.writer(file)
    writer.writerow(csv_heading)
    for items in training_data:
        writer.writerow(items)
    file.close()

    # Randomly shuffle the testing data and save into csv file
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


def convert_to_tf_lite(model: keras.models.Sequential) -> None:
    """ Converts the trained tensorflow keras model to a tensorflow lite model that can be implemented on
        microcontrollers

    :param model: the tensorflow model to be converted
    :return: None
    """
    # Convert the model to the TensorFlow Lite format with quantization
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    model_tflite = converter.convert()

    # Save the model to disk
    open('lite_model/model.tflite', "wb").write(model_tflite)


def get_dataset(filename: str) -> tuple:
    """ Reads data from a csv file and saves it into a pandas dataframe, which is then converted to a numpy array for
        use by the tensorflow neural network classifier

    :param filename: file name of the csv from which the data will be extracted from
    :return: tuple containing two numpy arrays corresponding to the data from the feature extraction and their
             corresponding labels
    """
    # Create a pandas dataframe from the data in the training or testing data
    emg_data = pd.read_csv(filename)

    # Create separate dataframe for the labels of the data
    emg_labels = emg_data.pop("Movement")

    # Convert both dataframes to numpy arrays to be used by the TensorFlow classifier
    emg_data = np.array(emg_data)
    emg_labels = np.array(emg_labels)

    return emg_data, emg_labels


def create_model() -> keras.models.Sequential:
    """ Creates the tensor flow neural network structure to be used to classify the emg data. Note that there are four
        features for each of the eight channels of the Mindrove armband, therefore the input layer should always have
        32 nodes. The number of nodes in the hidden layer can be adjusted according to performance requirements. The
        number of nodes in the output layer needs to correspond to the number of movements being classified

    :return: keras.models.Sequential, the neural network model created by the tensorFlow API
    """
    global movements
    # Create TensorFlow model structure
    model = keras.models.Sequential([
        keras.layers.Dense(32, input_shape=(32,)),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(len(movements)),
    ])

    # Set the "Loss" and "Optimiser" Functions
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)  # soft max
    optimizer = keras.optimizers.Adam(learning_rate=0.01)  # Learning Rate

    # Set metrics to be printed to the terminal
    metrics = ["accuracy"]

    # Create the model
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
    global epochs
    global batch_size
    # Build the model
    classifier_model = create_model()

    # Start the training of the neural network
    print("TRAINING".center(80, "-"))
    classifier_model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, shuffle=True, verbose=2)

    # Save the model
    classifier_model.get_weights()
    classifier_model.save_weights('classification_model/emg_classifier_weights.h5')

    # Save tf lite version of model for embedded systems
    convert_to_tf_lite(model=classifier_model)

    # Evaluate Model
    print("EVALUATING MODEL".center(80, "-"))
    classifier_model.evaluate(x_test, y_test, verbose=2)


"""
======================================= TEST CLASSIFICATION ALGORITHM =======================================
"""


def test_classifier(test_duration_seconds: int) -> None:
    """ Tests the TensorFlow neural network classifier on live data and prints the classifications to the terminal

    :param test_duration_seconds: duration for which testing should occur
    :return: None
    """
    # Create a new model and load previously saved weights from the trained model
    test_model = create_model()
    test_model.load_weights("classification_model/emg_classifier_weights.h5")

    # Initiate countdown to inform user that testing will begin
    timer_countdown(3, "EMG READING INITIATED")

    # Prepare board for data logging session (NOTE: The armband starts saving data from this point on ward)
    board = create_board()
    board.disable_board_logger()  # Removes annoying Mindrove Json object terminal logging :)
    emg_channels = board.get_emg_channels(BoardIds.MINDROVE_WIFI_BOARD)

    # Prepare Mindrove armband for data streaming
    board.prepare_session()
    while not board.is_prepared():
        pass

    # Start mindrove armband data stream
    board.start_stream()

    # Initiate timers
    time_start = time.perf_counter()
    time_end = time.perf_counter()

    while (time_end - time_start) < test_duration_seconds:
        # Wait until Mindrove armband has 200 raw emg samples
        data_on_board = board.get_board_data_count()
        while data_on_board < 200:
            data_on_board = board.get_board_data_count()

        # Get 200 samples of only the emg channels (channels 0 - 7)
        data = board.get_board_data(num_samples=200)[0:8]

        # For each channel of the EMG armband, apply filters to the raw emg data
        for channel in emg_channels:
            # Detrend data: remove linear offsets caused by the surface contact electrodes
            DataFilter.detrend(data=data[channel], detrend_operation=DetrendOperations.CONSTANT.value)
            # 5th order Bandpass Filter: 50Hz - 400Hz, see “Digital filtering of emg signals”, V. Zschorlich
            DataFilter.perform_bandpass(data=data[channel],
                                        sampling_rate=BoardShim.get_sampling_rate(BoardIds.MINDROVE_WIFI_BOARD),
                                        center_freq=225.0, band_width=350.0, order=5,
                                        filter_type=FilterTypes.BUTTERWORTH.value, ripple=0)
            # 1st order Highpass Filter: 0.5Hz cut-off (DC blocker). Mindrove recommendation
            DataFilter.perform_highpass(data=data[channel],
                                        sampling_rate=BoardShim.get_sampling_rate(BoardIds.MINDROVE_WIFI_BOARD),
                                        cutoff=0.5, order=1, filter_type=FilterTypes.BUTTERWORTH.value, ripple=0)

        # Get Features from data
        features = feature_extraction(sample_window=0.4, sample_range=200, channels=data[0:8])
        np_features = np.array([features])

        # Make prediction using Neural Network classifier
        val = np.argmax(test_model(np_features), axis=1)[0]

        # Decode and display prediction
        str_val = movement_decode(int(val))
        print(str_val)

        # Update Timer
        time_end = time.perf_counter()

    # End Mindrove armband streaming session
    board.stop_stream()
    board.release_session()
    print("Done")


"""
======================================= MAIN FUNCTION =======================================
"""


def main() -> None:
    """ Main loop of the python program. State machine that controls program flow

    :return: None
    """
    global movements
    global testing_time_seconds
    number_of_movements = len(movements)
    startup_print(startup=True)
    while True:
        print(''.center(80, '-'))

        user_in = input("Select an Option: ")

        if user_in == "1":
            print('CLASSIFIER TRAINING'.center(80, ' '))

            emg_data_acquisition(total_movements=number_of_movements)

            perform_feature_extraction()

            data_train, label_train = get_dataset(
                "feature_extraction/training.csv")
            data_test, label_test = get_dataset(
                "feature_extraction/testing.csv")

            print("Input data shape", data_train.shape, label_train.shape)

            train_classifier(data_train, label_train, data_test, label_test)

        elif user_in == "2":
            print('CLASSIFIER TESTING'.center(80, ' '))

            test_classifier(test_duration_seconds=testing_time_seconds)
        else:
            print("Closing program")
            quit()

        time.sleep(5)
        startup_print(startup=False)


if __name__ == "__main__":
    main()
