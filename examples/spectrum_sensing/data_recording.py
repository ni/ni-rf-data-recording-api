#
# Copyright 2025 Northeastern University and National Instruments Corporation
#
# SPDX-License-Identifier: MIT
#
import os
import glob
import json
import yaml
from typing import List, Tuple, Dict, Any

from lib import sync_settings
import main_rf_data_recording_api

global runFlag
runFlag = False

global usrp_init_status
usrp_init_status = False

api_pro = None


def get_usrp_init_status() -> bool:
    global usrp_init_status
    if sync_settings.start_rx_data_acquisition_called is True:
        usrp_init_status = False
    return usrp_init_status


def get_recording_status() -> bool:
    global runFlag
    runFlag = sync_settings.start_rx_data_acquisition_called
    return runFlag


def stop_rf_data_recording_api():
    sync_settings.external_stop_rx_data_acquisition_called = True


def delete_previous_records(rx_recorded_data_path: str, inference_results_folder_source: str):
    files = glob.glob(rx_recorded_data_path + '/*.sigmf-meta')
    # remove recorded data from previous records
    if files:
        print("Remove recorded data from previous record ...")
        files = sorted(files, key=os.path.getmtime)
        if len(files) > 5:
            for f_meta in files[5:]:
                os.remove(f_meta)
                filename = os.path.split(f_meta)[-1].split('.')[0]
                f_data = os.path.join(rx_recorded_data_path, filename) + '.sigmf-data'
                os.remove(f_data)
    # remove inference results from previous records
    files = glob.glob(inference_results_folder_source + "/*")
    if files:
        print("Remove inference results from previous records ...")
        files = sorted(files, key=os.path.getmtime)
        if len(files) > 2:
            for f in files[2:]:
                os.remove(f)


def run_ni_rf_data_recording_api(general_config: Dict[str, Any],
                                 txs_config: List[Dict[str, Any]],
                                 rxs_config: List[Dict[str, Any]],
                                 ni_rf_data_recording_api_path: str):
    # Add trailing slash if not exist
    ni_rf_data_recording_api_path = os.path.join(ni_rf_data_recording_api_path, '')
    tx1_config = txs_config[0]
    tx2_config = txs_config[1]
    if tx1_config["waveform_file_name"] == "OFF" and tx2_config["waveform_file_name"] == "OFF":
        default_rf_data_acq_config_file = "config_spectrum_sensing_1Rx.json"
        txs_config = []
    elif tx1_config["waveform_file_name"] != "OFF" and tx2_config["waveform_file_name"] != "OFF":
        default_rf_data_acq_config_file = "config_spectrum_sensing_2Tx_1Rx.json"
    elif tx1_config["waveform_file_name"] == "OFF":
        txs_config = [tx2_config]
        default_rf_data_acq_config_file = "config_spectrum_sensing_1Tx2_1Rx.json"
    elif tx2_config["waveform_file_name"] == "OFF":
        txs_config = [tx1_config]
        default_rf_data_acq_config_file = "config_spectrum_sensing_1Tx1_1Rx.json"
    else:
        raise Exception("Error: Invalid TX configuration.", tx1_config, tx2_config)

    # update default config
    _, updated_rf_data_acq_config_file = update_rf_api_config(
        default_rf_data_acq_config_file, general_config, txs_config, rxs_config,
        ni_rf_data_recording_api_path)

    # start rf data recording
    global usrp_init_status
    usrp_init_status = True
    # start main program
    rf_data_acq_config_file = updated_rf_data_acq_config_file
    main_rf_data_recording_api.main(rf_data_acq_config_file)


def update_rf_api_config(default_rf_data_acq_config_file: str, general_config: Dict[str, Any],
                         txs_config: List[Dict[str, Any]], rxs_config: List[Dict[str, Any]],
                         ni_rf_data_recording_api_path: str) -> Tuple[Dict[str, Any], str]:
    # Read default config
    rf_data_acq_config_file = default_rf_data_acq_config_file
    # Read RF Data config file extension
    _, extension = os.path.splitext(rf_data_acq_config_file)
    cfg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config")
    # read general parameter set from yaml config file
    if extension == ".yaml":
        with open(os.path.join(cfg_path, rf_data_acq_config_file), "r") as file:
            rf_data_acq_config = yaml.load(file, Loader=yaml.Loader)
    # Read general parameter set from json config file
    elif extension == ".json":
        with open(os.path.join(cfg_path, rf_data_acq_config_file), "r") as file:
            rf_data_acq_config = json.load(file)

    # Update Config
    for index in range(len(txs_config)):
        # Get new TX config
        tx_config = txs_config[index]
        waveform_file_name = tx_config["waveform_file_name"]
        std_key = waveform_file_name.split("_")[0]
        if std_key == "5GNR":
            tx_config["standard"] = "5gnr"
        elif std_key == "LTE":
            tx_config["standard"] = "lte"
        elif std_key == "Radar":
            tx_config["standard"] = "radar"
        elif std_key == "IEEE":
            tx_config["standard"] = "802.11"
        txs_config[index] = tx_config

    # Update general config
    rf_data_acq_config["general_config"]["nrecords"] = general_config["nrecords"]
    rf_data_acq_config["general_config"]["rx_recorded_data_path"] = \
        general_config["rx_recorded_data_path"]

    # Update TXs Config
    tx_variations_config_dict = rf_data_acq_config["transmitters_config"]
    if len(tx_variations_config_dict) != len(txs_config):
        raise Exception(
            "Error: Number of TX Stations on GUI should equal to them in default config file",
            len(tx_variations_config_dict), len(txs_config))
    for index in range(len(txs_config)):
        # Get new TX config
        tx_config = txs_config[index]
        # get default TX Config
        device_config = tx_variations_config_dict[index]
        parameters = device_config["Parameters"]
        # update Tx Config
        for key, value in tx_config.items():
            if key != "standard":
                parameters[key]["Values"] = tx_config[key]
            elif key == "standard":
                standard = value
                if standard == "5gnr":  # nr
                    parameters["waveform_generator"]["Values"] = "5gnr_ni_rfmx_rfws"
                    parameters["waveform_path"]["Values"] = "waveforms/nr/"
                    parameters["waveform_format"]["Values"] = "tdms"
                    # waveforms path
                    waveforms_path = os.path.join(
                        ni_rf_data_recording_api_path, "src", "waveforms", "nr")
                    waveform_list = glob.glob(waveforms_path + "/*")
                    # extract waveform name
                    waveform_list = list(
                        map(lambda x: os.path.split(x)[-1].split('.tdms')[0], waveform_list))
                    if tx_config["waveform_file_name"] in waveform_list:
                        parameters["waveform_file_name"]["Values"] = tx_config["waveform_file_name"]
                    else:
                        print("Warning: Selected waveform not avaliable",
                              tx_config["waveform_file_name"])
                        print("Warning: 5GNR Default waveform has been selected")
                        parameters["waveform_file_name"]["Values"] = \
                            "NR_FR1_DL_FDD_SISO_BW-10MHz_CC-1_SCS-30kHz_Mod-64QAM_OFDM_TM2"

                elif standard == "lte":  # lte
                    parameters["waveform_generator"]["Values"] = "lte_ni_rfmx_rfws"
                    parameters["waveform_path"]["Values"] = "waveforms/lte/"
                    parameters["waveform_format"]["Values"] = "tdms"
                    # waveforms path
                    waveforms_path = os.path.join(
                        ni_rf_data_recording_api_path, "src", "waveforms", "lte")
                    waveform_list = glob.glob(waveforms_path + "/*")
                    # extract waveform name
                    waveform_list = list(
                        map(lambda x: os.path.split(x)[-1].split('.tdms')[0], waveform_list))
                    if tx_config["waveform_file_name"] in waveform_list:
                        parameters["waveform_file_name"]["Values"] = tx_config["waveform_file_name"]
                    else:
                        print("Warning: Selected waveform not avaliable",
                              tx_config["waveform_file_name"])
                        print("Warning: LTE Default waveform has been selected")
                        parameters["waveform_file_name"]["Values"] = \
                            "LTE_FDD_DL_10MHz_CC-1_E-UTRA_E-TM2"
                elif standard == "radar":  # radar
                    parameters["waveform_generator"]["Values"] = "radar_nist"
                    parameters["waveform_path"]["Values"] = "waveforms/radar/"
                    parameters["waveform_format"]["Values"] = "matlab"
                    # waveforms path
                    waveforms_path = os.path.join(
                        ni_rf_data_recording_api_path, "src", "waveforms", "radar")
                    waveform_list = glob.glob(waveforms_path + "/*")
                    # extract waveform name
                    waveform_list = list(
                        map(lambda x: os.path.split(x)[-1].split('.mat')[0], waveform_list))
                    if tx_config["waveform_file_name"] in waveform_list:
                        parameters["waveform_file_name"]["Values"] = tx_config["waveform_file_name"]
                    else:
                        print("Warning: Selected waveform not avaliable",
                              tx_config["waveform_file_name"])
                        print("Warning: Radar Default waveform has been selected")
                        parameters["waveform_file_name"]["Values"] = "RadarWaveform_BW_2M"
                elif standard == "802.11":  # wifi
                    parameters["waveform_generator"]["Values"] = "802.11_ieee_gen_matlab"
                    parameters["waveform_path"]["Values"] = "waveforms/wifi/"
                    parameters["waveform_format"]["Values"] = "matlab_ieee"
                    # waveforms path
                    waveforms_path = os.path.join(
                        ni_rf_data_recording_api_path, "src", "waveforms", "wifi")
                    # For wifi they are currenty directories
                    folder_name_list = []
                    for it in os.scandir(waveforms_path):
                        if it.is_dir():
                            folder_list = os.path.split(it.path)
                            folder_name_list.append(folder_list[1])

                    if tx_config["waveform_file_name"] in folder_name_list:
                        parameters["waveform_file_name"]["Values"] = tx_config["waveform_file_name"]
                    else:
                        print("Warning: Selected waveform not avaliable",
                              tx_config["waveform_file_name"])
                        print("Warning: Wifi Default waveform has been selected")
                        parameters["waveform_file_name"]["Values"] = \
                            "IEEE_tx11ac_legacy_20MHz_80MSps_MCS7_27bytes_1frame"
                else:
                    raise Exception("Error: Invalid standard in TX config", standard)
        device_config["Parameters"] = parameters
        tx_variations_config_dict[index] = device_config
    rf_data_acq_config["transmitters_config"] = tx_variations_config_dict

    # Update RXs Config
    rxs_variations_config_dict = rf_data_acq_config["receivers_config"]
    if len(rxs_variations_config_dict) != len(rxs_config):
        raise Exception(
            "Error: Number of Rx Stations on GUI should equal to them in default config file",
            len(rxs_variations_config_dict), len(rxs_config))
    for index in range(len(rxs_config)):
        # Get new Rx config
        rx_config = rxs_config[index]
        # get default Rx Config
        rx_device_config = rxs_variations_config_dict[index]

        parameters = rx_device_config["Parameters"]

        # update rx Config
        for key, value in rx_config.items():
            parameters[key]["Values"] = rx_config[key]
        rx_device_config["Parameters"] = parameters
        rxs_variations_config_dict[index] = rx_device_config
    rf_data_acq_config["receivers_config"] = rxs_variations_config_dict

    # Remove comments contained in "possible_values"
    if "possible_values" in rf_data_acq_config.keys():
        del rf_data_acq_config["possible_values"]

    # Write result to JOSN Config file
    updated_rf_data_acq_config_file = os.path.join(cfg_path, "config.json")
    with open(updated_rf_data_acq_config_file, "w") as file:
        json.dump(rf_data_acq_config, file)

    return rf_data_acq_config, updated_rf_data_acq_config_file
