#
# Copyright 2025 Northeastern University and National Instruments Corporation
#
# SPDX-License-Identifier: MIT
#
from PIL import Image, ImageChops
import os
import glob
import numpy as np
from tqdm import tqdm


def trim(im):
    bg = Image.new(im.mode, im.size, im.getpixel((0, 0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -100)
    bbox = diff.getbbox()
    if bbox:
        return im.crop(bbox)


def image_cropper_file_based(source_path, destination_folder):
    filename = source_path.split('/')[-1]
    im = Image.open(source_path)

    cropped_im = trim(im)
    image_array = np.array(cropped_im)[:, 0:-1, :]
    cropped_im = Image.fromarray(image_array)

    # rotating a image 90 deg counter clockwise
    cropped_im = cropped_im.rotate(270, expand=1)

    cropped_path = os.path.abspath(destination_folder) + '/' + filename
    cropped_im.save(cropped_path)

    cropped_im.close()

    return cropped_path


def image_cropper(source_folder, destination_folder):

    # check source folder
    if not os.path.isdir(source_folder):
        raise Exception('ERROR: dataset_folder is not exist')

    # create destination folder if not existing
    if not os.path.isdir(destination_folder):
        print('Create destination folder: ' + destination_folder)
        os.makedirs(destination_folder)

    # find all files
    file_list = glob.glob(os.path.abspath(source_folder)+'/*')

    print('********** Cropping the white margines in spectrograms **********')

    for source_path in tqdm(file_list):
        image_cropper_file_based(source_path, destination_folder)
