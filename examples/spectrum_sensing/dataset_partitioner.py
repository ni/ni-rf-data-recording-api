#
# Copyright 2025 Northeastern University and National Instruments Corporation
#
# SPDX-License-Identifier: MIT
#
import os
import glob
import random
from typing import List


def dataset_partitioner(spectrogram_folder: str, train_val_test_split_rate: List[float]):

    if not os.path.isdir(spectrogram_folder):
        raise Exception('ERROR: spectrogram_folder is not exist')

    base_image_address, images_folder_name = os.path.split(os.path.abspath(spectrogram_folder))

    # check if default folder name is "images" otherwise Yolo will report error or you need
    # to change it.
    if images_folder_name != 'images':
        print("Warning: the name of images folder should be images, Yolo will report error")
        print(' Or you need to adopt Yolo code')

    all_images = glob.glob(spectrogram_folder + '/*')

    # print number of images
    n_images = len(all_images)
    print("Number of images", n_images)

    # randomize file order
    random.shuffle(all_images)
    print('********** Split Dataset **********')
    print('Split rate', train_val_test_split_rate)

    # split available files into training / validation and test data sets
    train_split = train_val_test_split_rate[0]
    val_split = train_val_test_split_rate[1]
    test_split = train_val_test_split_rate[2]  # noqa: F841

    training_paths = all_images[:int(train_split*n_images)]
    validation_paths = all_images[int(train_split*n_images):int((train_split+val_split)*n_images)]
    test_paths = all_images[int((train_split+val_split)*n_images):]

    destination_image_txt = os.path.join(base_image_address, 'training.txt')
    with open(destination_image_txt, 'w') as handle:
        for path in training_paths:
            # print(path, "training path")
            handle.write(path)
            handle.write('\n')
    destination_image_txt = os.path.join(base_image_address, 'validation.txt')
    with open(destination_image_txt, 'w') as handle:
        for path in validation_paths:
            # print(path, "validation path")
            handle.write(path)
            handle.write('\n')

    destination_image_txt = os.path.join(base_image_address, 'test.txt')
    with open(destination_image_txt, 'w') as handle:
        for path in test_paths:
            # print(path, "test path")
            handle.write(path)
            handle.write('\n')
