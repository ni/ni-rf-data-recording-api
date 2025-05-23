#
# Copyright 2025 Northeastern University and National Instruments Corporation
#
# SPDX-License-Identifier: MIT
#
import glob
import os
import time
from datetime import datetime

# import API functions
import image_cropper
import spectrogram_creator
import torch
# To print colours
from termcolor import colored


if __name__ == "__main__":
    # ------------------------------------------
    # ------------- Yolov5 Config --------------
    # ------------------------------------------
    weights_path = '/dev/ni-rf-data-recording-api/examples/spectrum_sensing/model/example.pt'
    YOLOv5_dir = '/dev/yolov5'
    path_base = "/dev/datasets/"

    device = 'cpu'  # for CPU use 'cpu', for GPU use '0' or just 0
    rx_recorded_data_path = path_base + "recorded_data/"
    spectrogram_folder = path_base + 'images/'
    inference_results_folder = path_base + 'results/'
    figure_size = [8, 12]
    figure_dpi = 100
    image_dims = (620, 925, 3)
    class_list = ['5GNR', 'LTE', 'Radar', '802.11']
    color_list = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (32, 165, 218)]
    line_thickness = 2
    conf = 0.4
    image_size = 640
    delete_analyzed_data = True

    # ------------------------------------------
    # ------------- Check paths ----------------
    # ------------------------------------------
    # ------------------------------------------
    # initialization
    # Add trailing slash if not exist
    rx_recorded_data_path = os.path.join(rx_recorded_data_path, '')
    spectrogram_folder = os.path.join(spectrogram_folder, '')
    inference_results_folder = os.path.join(inference_results_folder, '')

    if not os.path.isdir(rx_recorded_data_path):
        raise Exception('ERROR: Dataset_folder is not exist! ', rx_recorded_data_path)

    # create spectrogram folder if not existing
    if not os.path.isdir(spectrogram_folder):
        print('Create spectrogram folder: ' + spectrogram_folder)
        os.makedirs(spectrogram_folder)

    # create inference results folder if not existing
    if not os.path.isdir(inference_results_folder):
        print('Create inference results folder: ' + inference_results_folder)
        os.makedirs(inference_results_folder)

    # -------------------------------------------
    # ------------- Model Initialization  ------
    # ------------------------------------------
    model = torch.hub.load(YOLOv5_dir, source='local', model='custom', path=weights_path,
                           force_reload=False, autoshape=True, device=device)
    model.names = class_list
    model.conf = conf

    # Ctrl+C causes KeyboardInterrupt to be raised, just catch it outside the loop and ignore it.
    try:
        while True:
            # get list of all sigmf meta data files
            metadata_filelist = glob.glob(rx_recorded_data_path + '*' + '.sigmf-meta')
            # run only if the list is not empty
            if not metadata_filelist:
                time_stamp_micro_sec = datetime.now().strftime("%Y_%m_%d-%H_%M_%S_%f")
                time_stamp_milli_sec = time_stamp_micro_sec[:-3]
                print(f'{time_stamp_milli_sec} : Inference waiting for new recorded data')
                time.sleep(1)
            else:
                # sort the files by date/time in ascending order (oldest first)
                metadata_filelist = sorted(metadata_filelist, key=os.path.getmtime)
                # pick the oldest file and normalize to posix path
                metadata_file_path = os.path.normpath(metadata_filelist[0]).replace('\\', '/')

                # start time
                start_time = time.time()

                # create spectrogram image
                try:
                    spectrogram_creator.spectrogram_creator_file_based(
                        metadata_file_path, spectrogram_folder, figure_size, figure_dpi)
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    print(colored(f"Error: {metadata_file_path} - {type(e).__name__}", "red"))
                    continue
                finally:
                    if delete_analyzed_data:
                        binary_data_file = metadata_file_path.split('sigmf-meta')[0] + 'sigmf-data'
                        try:
                            # Try to delete the meta-data file.
                            os.remove(metadata_file_path)
                            # Try to delete the binary data file.
                            os.remove(binary_data_file)
                        except OSError as e:
                            print(colored(f"Error: {e.filename} - {e.strerror}.", "red"))

                # crop image to size
                filename = metadata_file_path.split('/')[-1].split('-meta')[0] + '.jpg'
                source_path = os.path.join(spectrogram_folder, filename)
                cropped_path = image_cropper.image_cropper_file_based(
                    source_path, spectrogram_folder)
                print(cropped_path)

                # run model inference
                results = model(cropped_path, size=image_size)
                # save results
                results.save(save_dir=inference_results_folder, exist_ok=True)

                if delete_analyzed_data:
                    try:
                        # Try to delete spectrogram image
                        os.remove(cropped_path)
                    except OSError as e:
                        print(colored(f"Error: {e.filename} - {e.strerror}.", "red"))

                # end inference time
                end_time = time.time()
                time_elapsed = end_time - start_time
                time_elapsed_ms = int(time_elapsed * 1000)
                print(
                    "Elapsed time of Data Inference per record:",
                    colored(time_elapsed_ms, "yellow"),
                    "ms",
                )

    except KeyboardInterrupt:
        pass
