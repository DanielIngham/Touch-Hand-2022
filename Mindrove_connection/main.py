"""
TOUCH PROSTHETICS - EMG CLASSIFICATION SYSTEM
    Author: Daniel Ingham
    Last Revision: 2022/08/12
    See
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import mindrove
from mindrove.board_shim import BoardShim, MindRoveInputParams
from mindrove.data_filter import DataFilter, FilterTypes, AggOperations, NoiseTypes


def main():
    # Enables MINDROVE terminal logging
    # BoardShim.enable_dev_board_logger()
    time.sleep(2)
    # The following parameters were found from: https://docs.mindrove.com/SupportedBoards.html#mindrove-arb-arc
    params = MindRoveInputParams()
    params.ip_port = 4210
    # params.ip_address = "192.168.4.1" # Does not work
    params.timeout = 10
    board_id = 0

    board = BoardShim(board_id, params)
    emg_channels = board.get_emg_channels(board_id)
    board_info = board.get_board_descr(board_id)

    print(board_info)

    # Prepare MINDROVE armband for data communication
    board.prepare_session()
    while not board.is_prepared():
        pass

    print("READY?")

    start_time = time.time()
    board.start_stream()

    stop_watch = time.time() - start_time
    flex_start = False

    while stop_watch < 3:
        if stop_watch > 1 and not flex_start:
            print("GO!")
            flex_start = True

        stop_watch = time.time() - start_time

    data = board.get_board_data()
    board.stop_stream()
    board.release_session()

    plt.figure()
    x_values = list(data[0])
    plt.plot(range(len(x_values)),x_values)
    plt.savefig('before_processing.png')


    # DataFilter.perform_bandstop(new_data, BoardShim.get_sampling_rate(board_id), 50.0, 1.0, 3,FilterTypes.BUTTERWORTH.value, 0)
    DataFilter.perform_bandpass(data[0], BoardShim.get_sampling_rate(board_id), 51.0, 100.0, 2,FilterTypes.BUTTERWORTH.value, 0)
    # DataFilter.perform_highpass(new_data, BoardShim.get_sampling_rate(board_id), 0.5, 2, FilterTypes.BUTTERWORTH.value,0)

    plt.figure()
    x_values = list(data[0])
    plt.plot(range(len(x_values)), x_values)
    plt.savefig('after_processing.png')




if __name__ == "__main__":
    main()
