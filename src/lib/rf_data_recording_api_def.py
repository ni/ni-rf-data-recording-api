#
# Copyright 2023 National Instruments Corporation
#
# SPDX-License-Identifier: MIT
#
"""
RF Data Recording API Main Class
"""
# Description:
#   This file is the main class of RF data recording API, where most of processing is executed.
#
# Pre-requests: Install UHD with Python API enabled
#

import os
import yaml
import signal

# from timeit import default_timer as timer
from pathlib import Path
import pandas as pd
import uhd
import math
import lib.run_rf_replay_data_transmitter

# to read tdms properties from rfws file
# https://www.datacamp.com/community/tutorials/python-xml-elementtree
from re import X
import xml.etree.cElementTree as ET

import time

# importing the threading module
import threading

# import related functions
from lib import run_rf_replay_data_transmitter
from lib import run_rf_data_recorder
from lib import sync_settings
from lib import rf_data_recording_config_interface
from lib.run_mmWave_device import get_device_type, get_device_name
from lib.data_format_conversion_lib import str2bool, str2list


class RFDataRecorderAPI:
    """Top-level RF Data Recorder API class"""

    def __init__(self, rf_data_acq_config_file):
        # read general parameter set from config file
        variations_map = rf_data_recording_config_interface.generate_rf_data_recording_configs(
            rf_data_acq_config_file
        )

        # Store them in the class
        self.variations_map = variations_map

    # Modulation schemes: lookup table as a constant dictionary:
    modulation_schemes = {"1": "BPSK", "2": "QPSK", "4": "16QAM", "6": "64QAM", "8": "256QAM"}
    RFmode = ["Tx", "Rx"]
    API_operation_modes = ["Tx-only", "Rx-only", "Tx-Rx"]

    # Define mmWave Array Antenna Class for RF Data Recording API Config Parameters
    class MMWaveAntennaArrayConfig:
        """mmWave Antenna Array Config class"""

        def __init__(self, iteration_config, usrp_id, rf_mode):
            device_id = usrp_id + "_" + "mmwave_antenna_array" + "_"
            self.rf_mode = rf_mode
            self.serial_number = iteration_config[device_id + "serial_number"]
            self.device_type = iteration_config[device_id + "device_type"]
            self.antenna_array_specification_table = iteration_config[device_id + "antenna_array_specification_table"]
            self.rf_frequency = iteration_config[device_id + "rf_frequency"]
            self.beamformer_config_mode = iteration_config[device_id + "beamformer_config_mode"]
            self.disabled_antenna_elements = str2list(iteration_config[device_id + "disabled_antenna_elements"], int)
            self.antenna_element_gain_list = str2list(iteration_config[device_id + "antenna_element_gain_list"], float)
            self.antenna_element_phase_list_deg = str2list(iteration_config[device_id + "antenna_element_phase_list_deg"], int)
            self.beam_gain_db = iteration_config[device_id + "beam_gain_db"]
            self.beam_angle_elevation_deg = int(iteration_config[device_id + "beam_angle_elevation_deg"])
            self.beam_angle_azimuth_deg = int(iteration_config[device_id + "beam_angle_azimuth_deg"])

    # Define mmWave Up Down Converter Class for RF Data Recording API Config Parameters
    class MMWaveUpDownConverterConfig:
        """mmWave Up Down Converter Config class"""

        def __init__(self, iteration_config, usrp_id):
            device_id = usrp_id + "_" + "mmwave_up_down_converter" + "_"
            self.serial_number = iteration_config[device_id + "serial_number"]
            self.num_channels = int(iteration_config[device_id + "num_channels"])
            '''
            Find the correct device_id of dual channel ud converter shared for both Tx and Rx in Tx config for Rx config
            Steps: 
            1. when the num_channels of ud converter is 2 in RX loop, check the number of parameters of ud converter
            2. if there are only two parameters exist: serial_number, num_channels, find the correct device_id
            '''
            if self.num_channels == 2 and usrp_id.startswith(RFDataRecorderAPI.RFmode[1]):
                device_id_prefix_columns = [col for col in iteration_config.index if col.startswith(device_id)]
                if len(device_id_prefix_columns) == 2:
                    device_id = self.mapping_with_tx_config(iteration_config, self.serial_number)
                    if device_id is None:
                        raise Exception(
                            f"ERROR: The serial number {self.serial_number} of ud converter in rx loop is different with tx.")
            self.device_type = iteration_config[device_id + "device_type"]
            self.if_frequency = iteration_config[device_id + "if_frequency"]
            self.rf_frequency = iteration_config[device_id + "rf_frequency"]
            self.lo_frequency = iteration_config[device_id + "lo_frequency"]
            self.bandwidth = iteration_config[device_id + "bandwidth"]
            self.disabled_channels = str2list(iteration_config[device_id + "disabled_channels"], int)
            self.enable_10MHz_clock_out = iteration_config[device_id + "enable_10MHz_clock_out"]
            self.enable_100MHz_clock_out = iteration_config[device_id + "enable_100MHz_clock_out"]
            self.clock_reference_100MHz = iteration_config[device_id + "clock_reference_100MHz"]
            self.enable_5V_out = iteration_config[device_id + "enable_5V_out"]
            self.enable_9V_out = iteration_config[device_id + "enable_9V_out"]

        # Get device_id of Tx ud converter containing extra config which is same with Rx via serial_number value
        def mapping_with_tx_config(self, iteration_config, serial_number):
            # get the list of columns of which the value is serial_number
            matching_indexes = iteration_config.index[iteration_config == serial_number].tolist()
            for index in matching_indexes:
                if index.startswith(RFDataRecorderAPI.RFmode[0]) and index.endswith("serial_number"):
                    start_index = index.rfind("serial_number")
                    device_id = index[:start_index]
                    return device_id
                else:
                    return None

    # Define TX Class for RF Data Reecording API Config Parameters
    class TxRFDataRecorderConfig:
        """Tx RFDataRecorder Config class"""

        def __init__(self, iteration_config, general_config, idx):
            # ============= TX Config parameters =============
            tx_id = RFDataRecorderAPI.RFmode[0] + str(idx)
            # Device args to use when connecting to the USRP, type=str",
            self.args = iteration_config[tx_id + "_args"]
            # "RF center frequency in Hz, type = float ",
            self.freq = iteration_config[tx_id + "_freq"]
            # lo_offset: type=float, help="LO offset in Hz")
            self.lo_offset = iteration_config[tx_id + "_lo_offset"]
            # enable_lo_offset: type=str2bool, Enable LO offset True or false")
            self.enable_lo_offset = iteration_config[tx_id + "_enable_lo_offset"]
            # "rate of radio block, type = float ",
            self.rate = iteration_config[tx_id + "_rate"]
            # "rate_source: pssoible options
            # (user_defined: given in variations section),
            # (waveform_config: read from waverform config properties)"
            self.rate_source = iteration_config[tx_id + "_rate_source"]
            # "gain for the RF chain, type = float",
            self.gain = iteration_config[tx_id + "_gain"]
            # "antenna selection, type = str",
            self.antenna = iteration_config[tx_id + "_antenna"]
            # "analog front-end filter bandwidth in Hz, type = float",
            self.bandwidth = iteration_config[tx_id + "_bandwidth"]
            # "waveform generator, type = str ",
            self.waveform_generator = iteration_config[tx_id + "_waveform_generator"]
            # "waveform file name, type = str ",
            self.waveform_file_name = iteration_config[tx_id + "_waveform_file_name"]
            # "path to TDMS/mat/... file, type = str ",
            self.waveform_path = iteration_config[tx_id + "_waveform_path"]
            # waveform_path_type: type of waveform path:relative or absolute
            # absolute: Full path to waveform should be given to use waveforms from another directory
            # relative: path related to waveform folder given with the API
            self.waveform_path_type = iteration_config[tx_id + "_waveform_path_type"]
            if self.waveform_path_type == "relative":
                dir_path = os.path.dirname(__file__)
                src_path = os.path.split(dir_path)[0]
                self.waveform_path = os.path.join(src_path, self.waveform_path)
            elif self.waveform_path_type == "absolute":
                pass
            else:
                raise Exception("Error: Unknow waveform path type", self.waveform_path_type)
            # "possible values: tdms, matlab_ieee, type = str ",
            self.waveform_format = iteration_config[tx_id + "_waveform_format"]
            # "clock reference source (internal, external, gpsdo, type = str",
            self.clock_reference = iteration_config["tx_clock_reference"]
            # "radio block to use (e.g., 0 or 1), type = int",
            self.radio_id = iteration_config["tx_radio_id"]
            # "radio channel to use, type = int",
            self.radio_chan = iteration_config["tx_radio_chan"]
            # "replay block to use (e.g., 0 or 1), type = int",
            self.replay_id = iteration_config["tx_replay_id"]
            # "replay channel to use, type = int ",
            self.replay_chan = iteration_config["tx_replay_chan"]
            # "duc id to use, type = int ",
            self.duc_id = iteration_config["tx_duc_id"]
            # "duc channel to use, type = int ",
            self.duc_chan = iteration_config["tx_duc_chan"]
            # "Hardware type, i.e. for USRP: USRP mboard ID (X310, or ....)
            self.hw_type = iteration_config[tx_id + "_hw_type"]
            # "Hardware subtype, i.e. for USRP: daughter board type (UBX-160, CBX-120)
            self.hw_subtype = iteration_config[tx_id + "_hw_subtype"]
            # "Hardware serial number, i.e. for USRP: mboard serial number
            self.seid = iteration_config[tx_id + "_seid"]
            # "HW RF maximum supported bandwidth
            self.max_RF_bandwidth = iteration_config[tx_id + "_max_RF_bandwidth"]
            # Define dictionary for tx waveform config
            waveform_config = {}
            self.waveform_config = waveform_config

            self.enable_mmwave = False
            if str2bool(general_config["enable_mmwave"]):
                self.enable_mmwave = True
                self.mmwave_antenna_array_parameters = RFDataRecorderAPI.MMWaveAntennaArrayConfig(
                    iteration_config,
                    tx_id,
                    RFDataRecorderAPI.RFmode[0])
                self.mmwave_up_down_converter_parameters = RFDataRecorderAPI.MMWaveUpDownConverterConfig(
                    iteration_config, tx_id)

    # Define RX Class for RX Config Parameters
    class RxRFDataRecorderConfig:
        """Rx RFDataRecorder Config class"""

        def __init__(self, iteration_config, general_config, idx):
            # ============= RX Config parameters =============
            rx_id = RFDataRecorderAPI.RFmode[1] + str(idx)
            # Device args to use when connecting to the USRP, type=str",
            self.args = iteration_config[rx_id + "_args"]
            # "RF center frequency in Hz, type = float ",
            self.freq = iteration_config[rx_id + "_freq"]
            # "rate of radio block, type = float ",
            self.rate = iteration_config[rx_id + "_rate"]
            # "rate_source: pssoible options
            # (user_defined: given in variations section),
            # (waveform_config: read from waverform config properties)"
            self.rate_source = iteration_config[rx_id + "_rate_source"]
            # "gain for the RF chain, type = float",
            self.gain = iteration_config[rx_id + "_gain"]
            # "rx bandwidth in Hz, type = float",
            self.bandwidth = iteration_config[rx_id + "_bandwidth"]
            # "radio channel to use, type = int",
            self.channels = iteration_config[rx_id + "_channels"]
            # "antenna selection, type = str",
            self.antenna = iteration_config[rx_id + "_antenna"]
            # "clock reference source (internal, external, gpsdo, type = str",
            self.clock_reference = iteration_config[rx_id + "_clock_reference"]
            # "time duration of IQ data acquisition"
            self.duration = iteration_config[rx_id + "_duration"]
            # "number of snapshots from RX IQ data acquisition"
            self.nrecords = general_config["nrecords"]
            # captured data file name
            self.captured_data_file_name = general_config["captured_data_file_name"]
            # "path to store captured rx data, type = str",
            self.rx_recorded_data_path = general_config["rx_recorded_data_path"]
            # rx recorded data saving format, type = str, possible values "SigMF"
            self.rx_recorded_data_saving_format = general_config["rx_recorded_data_saving_format"]
            # "Hardware type, i.e. for USRP: USRP mboard ID (X310, or ....)
            self.hw_type = iteration_config[rx_id + "_hw_type"]
            # "Hardware subtype, i.e. for USRP: daughter board type (UBX-160, CBX-120)
            self.hw_subtype = iteration_config[rx_id + "_hw_subtype"]
            # "Hardware serial number, i.e. for USRP: mboard serial number
            self.seid = iteration_config[rx_id + "_seid"]
            # "HW RF maximum supported bandwidth
            self.max_RF_bandwidth = iteration_config[rx_id + "_max_RF_bandwidth"]

            # initialize rx parameters
            self.num_rx_samps = 0
            self.coerced_rx_rate = 0.0
            self.coerced_rx_freq = 0.0
            self.coerced_rx_gain = 0.0
            self.coerced_rx_bandwidth = 0.0
            self.coerced_rx_lo_source = 0.0
            # channel parameters of this RX
            # expected channel atteuntion, type = float"
            self.channel_attenuation_db = iteration_config[rx_id + "_channel_attenuation_db"]

            self.enable_mmwave = False
            if str2bool(general_config["enable_mmwave"]):
                self.enable_mmwave = True
                self.mmwave_antenna_array_parameters = RFDataRecorderAPI.MMWaveAntennaArrayConfig(
                    iteration_config,
                    rx_id,
                    RFDataRecorderAPI.RFmode[1])
                self.mmwave_up_down_converter_parameters = RFDataRecorderAPI.MMWaveUpDownConverterConfig(
                    iteration_config, rx_id)

    ## Get Hw type, subtype and HW ID of TX and RX stations
    # For USRP:
    # HW type = USRP type, mboard ID, i.e. USRP X310, or ....
    # HW subtype = USRP daughterboard type
    # HW seid = USRP serial number
    # This extra step is a workaround to solve two limitations in UHD
    # For TX and RX: the master clock rate cannot be changed after opening the session
    # We need to know mBoard ID to select the proper master clock rate in advance
    # For TX based on RFNoc graph: Getting USRP SN is supported in Multi-USRP but not on RFNoC graph
    def get_hardware_info(self, variations_map, enable_console_logging: bool):
        variations_product = variations_map.variations_product
        general_config = variations_map.general_config

        def get_usrp_mboard_info(num_usrps, RFmode, variations_product):
            for n in range(num_usrps):
                idx = n + 1
                args_list = variations_product[RFmode + str(idx) + "_args"]
                args = args_list[0]
                # open the session to USRP
                usrp = uhd.usrp.MultiUSRP(args)
                # get USRP daughterboard ID, UBX, CBX ...etc
                if RFmode == RFDataRecorderAPI.RFmode[0]:
                    usrp_info = usrp.get_usrp_tx_info()
                    usrp_bandwidth = usrp.get_tx_bandwidth()
                    usrp_daughterboard_id = usrp_info["tx_id"]
                else:
                    usrp_info = usrp.get_usrp_rx_info()
                    usrp_daughterboard_id = usrp_info["rx_id"]
                    usrp_bandwidth = usrp.get_rx_bandwidth()
                # get USRP type, i.e. X310, or ....
                usrp_mboard_id = usrp_info["mboard_id"]
                temp = usrp_daughterboard_id.split(" ")
                usrp_daughterboard_id_wo_ref = temp[0]
                # get USRP serial number
                usrp_serial_number = usrp_info["mboard_serial"]

                if enable_console_logging:
                    print(RFmode, " USRP number ", idx, " info:")
                    print(
                        "usrp_mboard_id:",
                        usrp_mboard_id,
                        ", usrp_serial_number:",
                        usrp_serial_number,
                        ", usrp_daughterboard_id:",
                        usrp_daughterboard_id_wo_ref,
                        ", usrp_RF_bandwidth:",
                        usrp_bandwidth,
                    )

                data_frame_i = pd.DataFrame(
                    {RFmode + str(idx) + "_hw_type": ["USRP " + usrp_mboard_id]}
                )
                variations_product = variations_product.merge(data_frame_i, how="cross")

                data_frame_i = pd.DataFrame(
                    {RFmode + str(idx) + "_hw_subtype": [usrp_daughterboard_id_wo_ref]}
                )
                variations_product = variations_product.merge(data_frame_i, how="cross")

                data_frame_i = pd.DataFrame({RFmode + str(idx) + "_seid": [usrp_serial_number]})
                variations_product = variations_product.merge(data_frame_i, how="cross")

                data_frame_i = pd.DataFrame(
                    {RFmode + str(idx) + "_max_RF_bandwidth": [usrp_bandwidth]}
                )
                variations_product = variations_product.merge(data_frame_i, how="cross")

            return variations_product

        def add_mmwave_hw_type(RFmode, idx, variations_product, device_name):
            hw_type_index = RFmode + str(idx) + "_hw_type"
            variations_product[hw_type_index] = variations_product[hw_type_index] + " + " + device_name
            return variations_product

        def get_mmwave_device_info(num_usrps, RFmode, variations_product):
            for n in range(num_usrps):
                idx = n + 1
                # mmwave beam former
                device_id_beam_former = RFmode + str(idx) + "_mmwave_antenna_array_"
                serial_number_list_beam_former = variations_product[device_id_beam_former + "serial_number"]
                serial_number_beam_former = serial_number_list_beam_former[0]
                device_name_beam_former = get_device_name(serial_number_beam_former)
                # Add mmwave device name to hw_type
                variations_product = add_mmwave_hw_type(RFmode, idx, variations_product, device_name_beam_former)
                # Add device type
                device_type_beam_former = get_device_type(serial_number_beam_former)
                if device_type_beam_former is None:
                    raise Exception(
                        f"ERROR: The device type of beam former can't be found. SN: {serial_number_beam_former}")
                else:
                    data_frame_i = pd.DataFrame({device_id_beam_former + "device_type": [device_type_beam_former]})
                    variations_product = variations_product.merge(data_frame_i, how="cross")

                # mmwave ud converter
                device_id_ud_converter = RFmode + str(idx) + "_mmwave_up_down_converter_"
                serial_number_list_ud_converter = variations_product[device_id_ud_converter + "serial_number"]
                serial_number_ud_converter = serial_number_list_ud_converter[0]
                device_name_ud_converter = get_device_name(serial_number_ud_converter)
                # Add mmwave device name to hw_type
                variations_product = add_mmwave_hw_type(RFmode, idx, variations_product,
                                                        device_name_ud_converter)
                # Add device type
                num_channels = int(variations_product[device_id_ud_converter + "num_channels"][0])
                if RFmode == RFDataRecorderAPI.RFmode[1] and num_channels == 2:
                    device_id_prefix_columns = [col for col in variations_product.columns if
                                                col.startswith(device_id_ud_converter)]
                    # no need to find device type for rx dual udc which has only 2 params
                    if len(device_id_prefix_columns) == 2:
                        continue
                device_type_ud_converter = get_device_type(serial_number_ud_converter)
                if device_type_ud_converter is None:
                    raise Exception(
                        f"ERROR: The device type of ud converter can't be found. SN: {serial_number_ud_converter}")
                else:
                    data_frame_i = pd.DataFrame({device_id_ud_converter + "device_type": [device_type_ud_converter]})
                    variations_product = variations_product.merge(data_frame_i, how="cross")
            return variations_product

        # get hW info of TX Stations
        num_tx_usrps = int(general_config["num_tx_usrps"])
        if num_tx_usrps > 0:
            # if Tx station is USRP
            variations_product = get_usrp_mboard_info(
                num_tx_usrps, RFDataRecorderAPI.RFmode[0], variations_product
            )
            if str2bool(general_config["enable_mmwave"][0]):
                variations_product = get_mmwave_device_info(
                    num_tx_usrps, RFDataRecorderAPI.RFmode[0], variations_product)

        # get hW info of RX Stations
        num_rx_usrps = int(general_config["num_rx_usrps"])
        if num_rx_usrps > 0:
            # if Rx station is USRP
            variations_product = get_usrp_mboard_info(
                num_rx_usrps, RFDataRecorderAPI.RFmode[1], variations_product
            )
            if str2bool(general_config["enable_mmwave"][0]):
                variations_product = get_mmwave_device_info(
                    num_rx_usrps, RFDataRecorderAPI.RFmode[1], variations_product
                )

        if enable_console_logging:
            print("Updated variations cross product including Tx and RX stations HW info: ")
            print(variations_product)

        # Store variations product in its class
        variations_map.variations_product = variations_product

        return variations_map

    ## Update TX rate based on user selection:
    # "waveform_config": Set TX rate and bandwith based on waveform config
    # "user_defined": use the given value by the user in the config file: config_rf_data-recording_api
    def update_tx_rate(txs_data_recording_api_config):

        # Start for loop to update each TX config based on its rate config source
        for tx_idx, tx_data_recording_api_config in enumerate(txs_data_recording_api_config):
            if tx_data_recording_api_config.rate_source == "waveform_config":
                tx_data_recording_api_config.bandwidth = (
                    tx_data_recording_api_config.waveform_config["bandwidth"]
                )
                tx_data_recording_api_config.rate = tx_data_recording_api_config.waveform_config[
                    "sample_rate"
                ]
            txs_data_recording_api_config[tx_idx] = tx_data_recording_api_config

        return txs_data_recording_api_config

    ## Update RX rate based on user selection:
    # "waveform_config": Set RX rate and bandwith based on waveform config
    #                    In case of multi Tx: Use the max bandwith and max rate of TX USRPs as a reference on RX
    # "user_defined": use the given value by the user in the config file: config_rf_data-recording_api
    def update_rx_rate(rxs_data_recording_api_config, txs_data_recording_api_config):

        # Initialize max rate and bandwidth
        max_tx_bandwidth = 0.0
        max_tx_rate = 0.0
        # Start for loop to update each RX based on its rate config source
        for rx_idx, rx_data_recording_api_config in enumerate(rxs_data_recording_api_config):
            if rx_data_recording_api_config.rate_source == "waveform_config":
                # find max rate and bandwidth of all TX USRPs
                # Do this check just for first RX USRP
                # The max values of TX usrps are valid for each Rx if it is configured to use "waveform_config"
                if rx_idx == 0:
                    for tx_idx, tx_data_recording_api_config in enumerate(
                        txs_data_recording_api_config
                    ):
                        if tx_data_recording_api_config.bandwidth > max_tx_bandwidth:
                            max_tx_bandwidth = tx_data_recording_api_config.bandwidth
                        if tx_data_recording_api_config.rate > max_tx_rate:
                            max_tx_rate = tx_data_recording_api_config.rate
                rx_data_recording_api_config.bandwidth = max_tx_bandwidth
                rx_data_recording_api_config.rate = max_tx_rate
            rxs_data_recording_api_config[rx_idx] = rx_data_recording_api_config

        return rxs_data_recording_api_config

    ## Update TX and RX rate based on user selection:
    # "waveform_config": Set TX or RX rate and bandwith based on waveform config
    #                    In case of multi Tx: Use the max bandwith and max rate of TX USRPs as a reference on RX
    # "user_defined": use the given value by the user in the config file: config_rf_data-recording_api
    def update_rate(self, txs_data_recording_api_config, rxs_data_recording_api_config):
        # First: Update TX Rate
        txs_data_recording_api_config = RFDataRecorderAPI.update_tx_rate(
            txs_data_recording_api_config
        )
        # Then: Update RX rate
        rxs_data_recording_api_config = RFDataRecorderAPI.update_rx_rate(
            rxs_data_recording_api_config, txs_data_recording_api_config
        )

        return txs_data_recording_api_config, rxs_data_recording_api_config

    ## Find proper master clock rate
    def calculate_master_clock_rate(requested_rate, usrp_mboard_id, args_in):
        def round_up_to_even(f):
            return math.ceil(f / 2.0) * 2

        # Derive master clock rate for X310 / X300 USRP
        if "X3" in usrp_mboard_id:
            # There are two master clock rates (MCR) supported on the X300 and X310: 200.0 MHz and 184.32 MHz.
            # The achievable sampling rates are by using an even decimation factor
            master_clock_rate_x310 = [200e6, 184.32e6]

            # Calculate the ratio between MCR and requested sampling rate
            ratio_frac = [x / requested_rate for x in master_clock_rate_x310]
            # Change to integer
            ratio_integ = [round(x) for x in ratio_frac]
            # Find the higher even number
            ratio_even = []
            for index, value in enumerate(ratio_frac):
                value = ratio_integ[index]
                if value < 1:
                    ratio_even.append(2)
                elif value < 2:
                    ratio_even.append(0)
                else:
                    ratio_even.append(round_up_to_even(value))
            # Calculate the deviation
            ratio_dev = []
            for index, value in enumerate(ratio_even):
                value1 = ratio_even[index]
                value2 = ratio_frac[index]
                ratio_dev.append(abs(value1 - value2))
            # Find the best MCR for the requested rate
            pos = ratio_dev.index(min(ratio_dev))
            master_clock_rate_config = master_clock_rate_x310[pos]
            if master_clock_rate_config == 184.32e6:
                args_out = args_in + ",master_clock_rate=184.32e6"
            else:
                args_out = args_in + ",master_clock_rate=200e6"
        # Derive master clock rate for X410 USRP
        elif "X4" in usrp_mboard_id:
            # There are two master clock rates (MCR) supported on the X4XX: 245.76 MHz and 250 MHz.
            # The achievable sampling rates are by using an even decimation factor
            master_clock_rate_x310 = [245.76e6, 250e6]

            # Calculate the ratio between MCR and requested sampling rate
            ratio_frac = [x / requested_rate for x in master_clock_rate_x310]
            # Change to integer
            ratio_integ = [round(x) for x in ratio_frac]
            # Find the higher even number
            ratio_even = []
            for index, value in enumerate(ratio_frac):
                value = ratio_integ[index]
                if value < 1:
                    ratio_even.append(2)
                elif value < 2:
                    ratio_even.append(0)
                else:
                    ratio_even.append(round_up_to_even(value))
            # Calculate the deviation
            ratio_dev = []
            for index, value in enumerate(ratio_even):
                value1 = ratio_even[index]
                value2 = ratio_frac[index]
                ratio_dev.append(abs(value1 - value2))
            # Find the best MCR for the requested rate
            pos = ratio_dev.index(min(ratio_dev))
            master_clock_rate_config = master_clock_rate_x310[pos]
            if master_clock_rate_config == 245.76e6:
                args_out = args_in + ",master_clock_rate=245.76e6"
            else:
                args_out = args_in + ",master_clock_rate=250e6"
        # Derive master clock rate for other USRPs is not supported yet
        else:
            print(
                "Warning: The code can derive the master clock rate for X410/X310/X300 USRPs only."
            )
            print("         The default master clock rate will be used.")
            args_out = args_in

        return args_out

    # Calculate USRP master clockrate based on given rate
    def find_proper_master_clock_rate(
        self, txs_data_recording_api_config, rxs_data_recording_api_config
    ):
        # Find master clock rate for TX USRPs
        for tx_idx, tx_data_recording_api_config in enumerate(txs_data_recording_api_config):
            tx_data_recording_api_config.args = RFDataRecorderAPI.calculate_master_clock_rate(
                tx_data_recording_api_config.rate,
                tx_data_recording_api_config.hw_type,
                tx_data_recording_api_config.args,
            )
            txs_data_recording_api_config[tx_idx] = tx_data_recording_api_config

        # Find master clock rate for RX USRPs
        for rx_idx, rx_data_recording_api_config in enumerate(rxs_data_recording_api_config):
            rx_data_recording_api_config.args = RFDataRecorderAPI.calculate_master_clock_rate(
                rx_data_recording_api_config.rate,
                rx_data_recording_api_config.hw_type,
                rx_data_recording_api_config.args,
            )
            rxs_data_recording_api_config[rx_idx] = rx_data_recording_api_config

        return txs_data_recording_api_config, rxs_data_recording_api_config

    # print iteration config
    def print_iteration_config(
        self,
        iteration_config,
        general_config,
        txs_data_recording_api_config,
        rxs_data_recording_api_config,
    ):
        import warnings

        warnings.filterwarnings("ignore")
        for tx_idx, tx_data_recording_api_config in enumerate(txs_data_recording_api_config):
            iteration_config[
                RFDataRecorderAPI.RFmode[0] + str(tx_idx + 1) + "_sample_rate"
            ] = tx_data_recording_api_config.rate
            iteration_config[
                RFDataRecorderAPI.RFmode[0] + str(tx_idx + 1) + "_bandwidth"
            ] = tx_data_recording_api_config.bandwidth

        for rx_idx, rx_data_recording_api_config in enumerate(rxs_data_recording_api_config):
            iteration_config[
                RFDataRecorderAPI.RFmode[1] + str(rx_idx + 1) + "_sample_rate"
            ] = rx_data_recording_api_config.rate
            iteration_config[
                RFDataRecorderAPI.RFmode[1] + str(rx_idx + 1) + "_bandwidth"
            ] = rx_data_recording_api_config.bandwidth
        warnings.filterwarnings("default")
        print("Iteration config: ")
        print(iteration_config)
        print("General config: ")
        print(general_config)

    ## Use Ctrl-handler to stop TX in case of Tx Only
    def call_stop_tx_siganl():
        # Ctrl+C handler
        def signal_handler(sig, frame):
            print("Exiting . . .")
            sync_settings.stop_tx_signal_called = True

        # Wait until the Tx station is ready and then print how to stop tx
        while sync_settings.start_rx_data_acquisition_called == False:
            time.sleep(0.1)  # sleep for 100ms

        # ** Wait until user says to stop **
        # Setup SIGINT handler (Ctrl+C)
        signal.signal(signal.SIGINT, signal_handler)
        list = ["\\", "|", "/", "—"]
        print("")
        print("Press Ctrl+C to stop RF streaming for this iteration ...")
        while sync_settings.stop_tx_signal_called == False:
            for i in range(0, 4):
                index = i % 4
                print("\rRF streaming {}".format(list[index]), end="")
                time.sleep(0.1)  # sleep for 100ms

    ## Start execution - TX emitters in parallel
    def start_execution_txs_in_parallel(
        self,
        txs_data_recording_api_config,
        rxs_data_recording_api_config,
        api_operation_mode,
        general_config,
        rx_data_nbytes_que,
    ):

        # initialize threads
        threads = []
        # start transmitters
        for idx, tx_data_recording_api_config in enumerate(txs_data_recording_api_config):
            process = threading.Thread(
                target=run_rf_replay_data_transmitter.rf_replay_data_transmitter,
                args=(txs_data_recording_api_config[idx],),
            )
            process.start()
            threads.append(process)

        # start receivers
        # Trigger Rx:
        # ---------The data acquisition will start as soon as one Tx station is ready and started signal transmission
        # ---------The flag: "sync_settings.start_rx_data_acquisition_called" is used for that
        # Stop Tx:
        # ------ As soon as the Rx data is recorded, the txs will stop data tranmission
        # ------ The flag "sync_settings.stop_tx_signal_called" is used for that
        for idx, rx_data_recording_api_config in enumerate(rxs_data_recording_api_config):
            process = threading.Thread(
                target=run_rf_data_recorder.rf_data_recorder,
                args=(
                    rxs_data_recording_api_config[idx],
                    txs_data_recording_api_config,
                    general_config,
                    rx_data_nbytes_que,
                ),
            )
            process.start()
            threads.append(process)

        # For Tx-only mode: the stop tx signal is done manaully using Ctrl+C command
        if api_operation_mode == RFDataRecorderAPI.API_operation_modes[0]:
            # Ctrl+C handler
            RFDataRecorderAPI.call_stop_tx_siganl()

        # We now pause execution on the main thread by 'joining' all of our started threads.
        # This ensures that each has finished processing the urls.
        for process in threads:
            process.join()

        # settling time
        time.sleep(0.05)

    ## Start execution - TX emitters in sequential
    def start_execution_txs_in_sequential(
        self,
        txs_data_recording_api_config,
        rxs_data_recording_api_config,
        api_operation_mode,
        general_config,
        rx_data_nbytes_que,
        enable_console_logging,
    ):

        for tx_idx, tx_data_recording_api_config in enumerate(txs_data_recording_api_config):
            ##  Initlize sync settings
            sync_settings.init()
            if enable_console_logging:
                print(
                    "Sync Status: Start Data acquisition called = ",
                    sync_settings.start_rx_data_acquisition_called,
                    " Stop Tx Signal called = ",
                    sync_settings.stop_tx_signal_called,
                )
            # initialize threads
            threads = []
            # start transmitter
            process = threading.Thread(
                target=run_rf_replay_data_transmitter.rf_replay_data_transmitter,
                args=(txs_data_recording_api_config[tx_idx],),
            )
            process.start()
            threads.append(process)

            # start receivers
            # Trigger Rx:
            # ---------The data acquisition will start as soon as one Tx station is ready and started signal transmission
            # ---------The flag: "sync_settings.start_rx_data_acquisition_called" is used for that
            # Stop Tx:
            # ------ As soon as the Rx data is recorded, the txs will stop data tranmission
            # ------ The flag "sync_settings.stop_tx_signal_called" is used for that

            # Read tx config of related Tx to be stored in meta-data
            # In case of squential tranmissions, store only the meta-data of active Tx.
            txs_data_recording_api_config_i = [txs_data_recording_api_config[tx_idx]]
            for rx_idx, rx_data_recording_api_config in enumerate(rxs_data_recording_api_config):
                rx_data_recording_api_config = rxs_data_recording_api_config[rx_idx]
                process = threading.Thread(
                    target=run_rf_data_recorder.rf_data_recorder,
                    args=(
                        rx_data_recording_api_config,
                        txs_data_recording_api_config_i,
                        general_config,
                        rx_data_nbytes_que,
                    ),
                )
                process.start()
                threads.append(process)

            # Tx-only mode
            if api_operation_mode == RFDataRecorderAPI.API_operation_modes[0]:
                # Ctrl+C handler
                RFDataRecorderAPI.call_stop_tx_siganl(api_operation_mode)

            # We now pause execution on the main thread by 'joining' all of our started threads.
            # This ensures that each has finished processing the urls.
            for process in threads:
                process.join()

            # settling time
            time.sleep(0.05)

    ## Execute Rxs only for RX only mode
    def start_rxs_execution(
        self,
        txs_data_recording_api_config,
        rxs_data_recording_api_config,
        general_config,
        rx_data_nbytes_que,
    ):

        threads = []
        # For Rx only, no trigger required from Tx to start data acquisition
        # Send a command to start RX data acquisition
        sync_settings.start_rx_data_acquisition_called = True

        for idx, rx_data_recording_api_config in enumerate(rxs_data_recording_api_config):
            process = threading.Thread(
                target=run_rf_data_recorder.rf_data_recorder,
                args=(
                    rxs_data_recording_api_config[idx],
                    txs_data_recording_api_config,
                    general_config,
                    rx_data_nbytes_que,
                ),
            )
            process.start()
            threads.append(process)

        # We now pause execution on the main thread by 'joining' all of our started threads.
        # This ensures that each has finished processing the urls.
        for process in threads:
            process.join()

        # settling time
        time.sleep(0.05)
