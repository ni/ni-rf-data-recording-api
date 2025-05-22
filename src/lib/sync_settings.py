#
# Copyright 2022 National Instruments Corporation
#
# SPDX-License-Identifier: MIT
#
"""
TX and Rx Data Acquisition Sync
"""
# Description:
#   define global variables to sync between Tx and Rx start and stop execution
#


def init():
    # start Rx data acquisition if TX signal is on the air
    global start_rx_data_acquisition_called
    start_rx_data_acquisition_called = False

    # stop TX data transmission if RX data acquisition is done
    global stop_tx_signal_called
    stop_tx_signal_called = False

    # stop API external
    global external_stop_rx_data_acquisition_called
    external_stop_rx_data_acquisition_called = False
