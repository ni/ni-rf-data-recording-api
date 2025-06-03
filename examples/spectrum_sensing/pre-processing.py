#
# Copyright 2025 Northeastern University and National Instruments Corporation
#
# SPDX-License-Identifier: MIT
#
import os
# import API functions
import spectrogram_creator
import image_cropper
import dataset_partitioner
import label_maker

if __name__ == "__main__":
    # ------------------------------------------
    # ------------- Configuration --------------
    # ------------------------------------------
    # specify folder for RF data recordings
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    datasets_path = os.path.join(curr_dir, 'datasets')
    rx_records_path = os.path.join(datasets_path, 'records')
    spectrogram_folder = os.path.join(datasets_path, 'images')
    images_bounding_boxes_folder = os.path.join(datasets_path, 'images_bounding_boxes')
    label_folder = os.path.join(datasets_path, 'labels')
    # ------------------------------------------
    train_val_test_split_rate = [0.9, 0.05, 0.05]
    figure_size = [8, 12]
    figure_dpi = 100
    image_dims = (620, 925, 3)
    class_list = ['5gnr', 'lte', 'radar', '802.11']
    color_list = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (32, 165, 218)]
    line_thickness = 2
    enable_training_mode = True
    # ------------------------------------------

    # Create Spectrograms
    spectrogram_creator.spectrogram_creator(
        rx_records_path, spectrogram_folder, figure_size, figure_dpi)

    # Cropping the white margines in spectrogram
    # provide source and destination paths
    image_cropper.image_cropper(spectrogram_folder, spectrogram_folder)

    if enable_training_mode:
        # dataset partitioning
        dataset_partitioner.dataset_partitioner(spectrogram_folder, train_val_test_split_rate)

        # create labels
        label_maker.label_maker(rx_records_path, spectrogram_folder, label_folder,
                                image_dims, class_list, color_list,
                                images_bounding_boxes_folder, line_thickness)
