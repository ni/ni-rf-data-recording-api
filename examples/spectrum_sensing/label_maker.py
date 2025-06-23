#
# Copyright 2025 Northeastern University and National Instruments Corporation
#
# SPDX-License-Identifier: MIT
#
import os
import glob
import cv2  # for reading images
import numpy as np
from typing import List, Tuple
from tqdm import tqdm
from sigmf import SigMFFile, sigmffile


def draw_boxes(boxes: List[List], image: cv2.typing.MatLike, write_path: str,
               class_list: List[str], image_dims: Tuple[int],
               color_list: List[Tuple[int, int, int]], line_thickness: int):
    ymin = 0
    ymax = image_dims[0] - 1
    for box in boxes:
        label = class_list.index(box[0])
        xmin = box[1]
        xmax = box[2]
        cv2.rectangle(img=image, pt1=(xmin, ymin), pt2=(xmax, ymax),
                      color=color_list[label], thickness=line_thickness)
    cv2.imwrite(write_path, np.uint8(image))


def txt_label_generator(object_name_list: List[str], object_boundary_list: List[List[int]],
                        class_list: List[str], image_dims: Tuple[int]) -> List[str]:
    image_h, image_w = image_dims[0], image_dims[1]
    txt_rows = []
    for object_label, boundary in zip(object_name_list, object_boundary_list):
        class_index = class_list.index(object_label)
        object_low = int(boundary[0])
        object_high = int(boundary[1])
        object_top = 0
        object_bottom = int(image_h - 1)
        x_center = 1.0 * (object_low + object_high) / (2 * image_w)
        y_center = 1.0 * (object_top + object_bottom) / (2 * image_h)
        width = 1.0 * (object_high - object_low) / image_w
        height = 1.0 * (object_bottom - object_top) / image_h
        this_row = str(class_index) + ' ' + str(x_center) + ' ' + str(y_center) + ' ' \
            + str(width) + ' ' + str(height)
        txt_rows.append(this_row)
    return txt_rows


def label_maker(dataset_folder: str, spectrogram_folder: str, label_folder: str,
                image_dims: Tuple[int], class_list: List[str],
                color_list: List[Tuple[int, int, int]],
                images_bounding_boxes_folder: str, line_thickness: int):

    print('********** Make Labels **********')
    # create spectrogram folder if not existing
    if not os.path.isdir(label_folder):
        print('Create labels folder:', label_folder)
        os.makedirs(label_folder)

    # creat folder for images bounding boxes if not existing
    if not os.path.isdir(images_bounding_boxes_folder):
        print('Create images_bounding_boxes folder:', images_bounding_boxes_folder)
        os.makedirs(images_bounding_boxes_folder)

    # get list of all sigmf meta data files
    metadata_filelist = glob.glob(dataset_folder + '/*.sigmf-meta')
    # get image file path
    image_list = glob.glob(spectrogram_folder + '/*')
    # extract image name
    image_list = list(map(lambda x: os.path.split(x)[-1].split('.jpg')[0], image_list))

    for metadata_path in tqdm(metadata_filelist):
        image_name = os.path.split(metadata_path)[-1].split('-meta')[0]

        # check if image name is within image list
        if image_name in image_list:
            object_name_list = []
            object_boundary_list = []
            boxes = []
            # Read SigMF meta-data and extract sampling rate
            metadata = sigmffile.fromfile(metadata_path)
            sample_rate_rx = metadata.get_global_field(SigMFFile.SAMPLE_RATE_KEY)
            frequency_per_pixel = sample_rate_rx / image_dims[1]

            # Get capture info associated with the start of annotation
            annotations = metadata.get_annotations()
            # we are using single annotation
            annotation = annotations[0]
            # Get Rx frequency
            annotation_start_idx = annotation[SigMFFile.START_INDEX_KEY]
            capture = metadata.get_capture_info(annotation_start_idx)
            freq_center_rx = capture.get(SigMFFile.FREQUENCY_KEY, 0)

            # get number of transmitters
            num_transmitters = annotation.get('num_transmitters')
            txs_config = annotation.get('system_components:transmitter')
            if num_transmitters != len(txs_config):
                raise Exception(
                    'ERROR: num_transmitters is not equal to the transmitters config class',
                    num_transmitters, len(txs_config))

            # iterate over different tranmitters to get their info
            for tx_index in range(num_transmitters):
                tx_config = txs_config[tx_index]
                # get standard
                standard = tx_config['signal:detail']['standard']
                # the training is limited to given class list
                if standard in class_list:
                    # get signal bandwidth
                    signal_bandwidth = tx_config['signal:detail'][standard]['bandwidth']

                    # get tx frequency
                    freq_center_tx = tx_config['signal:emitter']['frequency']
                    fmin = 1.0 * sample_rate_rx / 2 - 1.0 * signal_bandwidth / 2 \
                        - (freq_center_rx - freq_center_tx)
                    fmax = 1.0 * sample_rate_rx / 2 + 1.0 * signal_bandwidth / 2 \
                        - (freq_center_rx - freq_center_tx)
                    if fmin < 0 or fmax < 0:
                        raise Exception(
                            "Fmin and Fmax of the given waveform should be captured Rx spectrogram",
                            image_name, fmin, fmax)
                    xmin = int(1.0 * fmin / frequency_per_pixel)
                    xmax = int(1.0 * fmax / frequency_per_pixel)

                    object_name_list.append(standard)
                    object_boundary_list.append([xmin, xmax])
                    boxes.append([standard, xmin, xmax])

            # write labels
            if len(object_boundary_list) != 0:
                # We have objects in this image, generate the meta data:
                txt_row_list = txt_label_generator(
                    object_name_list, object_boundary_list, class_list, image_dims)
                label_path = label_folder + image_name + '.txt'

                with open(label_path, 'w') as handle:
                    for line in txt_row_list:
                        handle.write(line)
                        handle.write('\n')

            # sanity check / draw boxes
            if len(object_boundary_list) != 0:
                image_path = os.path.join(spectrogram_folder, image_name + '.jpg')
                this_image = cv2.imread(image_path)
                write_path = os.path.join(images_bounding_boxes_folder, image_name + '.jpg')

                draw_boxes(boxes, this_image, write_path, class_list,
                           image_dims, color_list, line_thickness)
