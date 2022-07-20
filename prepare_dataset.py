# -*- coding: utf-8 -*-
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.

import argparse
import os
import sys
import logging

import h5py
import json
import numpy as np

from typing import Dict, List
from collections import defaultdict, Counter
from PIL import Image


logger = logging.getLogger(name=__name__)

LABEL_DELIMITER = '|'   # Labels will be joined by delimiter and saved to hdf5
LOG_INTERVAL = 500  # Log the preprocessing progress every interval steps


def load_user_metadata(labels_file: str) -> Dict[str, List]:
    """
    Load labels and metadata keyed by `user_id`.

    :param labels_file:
        A .json file with a list of labels and metadata dictionaries. Each
        dictionary has keys: `[image_id,user_id,labels,fine_grained_labels]`.
        * `image_id` is the ID of an image.
        * `user_id` is the ID of the user `image_id` belongs to.
        * `labels` is a list of 17 higher-order class labels.
        * `fine_grained_labels` is a list of 1,628 fine-grained class labels.
    :return:
        A dictionary where key is `user_id` and value is a list of labels and
        metadata for each image `user_id` owns.
    """
    user_metadata = defaultdict(list)
    with open(labels_file) as f:
        metadata_list = json.load(f)

    for metadata in metadata_list:
        user_metadata[metadata["user_id"]].append(metadata)
    return user_metadata


def preprocess_federated_dataset(image_dir: str,
                                 labels_file: str,
                                 output_file: str):
    """
    Process images and labels into a HDF5 federated dataset where data is
    first split by train/test partitions and then split again by user ID.

    :param image_dir:
        Path to directory of images output from the script
        `download_dataset.sh`.
    :param labels_file:
        A .json file with a list of labels and metadata dictionaries. Each
        dictionary has keys: `[image_id,user_id,labels,fine_grained_labels]`.
        * `image_id` is the ID of an image.
        * `user_id` is the ID of the user `image_id` belongs to.
        * `labels` is a list of 17 higher-order class labels.
        * `fine_grained_labels` is a list of ~1,600 fine-grained class labels.
    :param output_file:
        Output path for HDF5 file. Use the postfix `.hdf5`.
    """
    logger.info('Preprocessing federated dataset.')
    user_metadata = load_user_metadata(labels_file)
    label_counter = Counter()
    fine_grained_label_counter = Counter()
    with h5py.File(output_file, 'w') as h5file:
        # Iterate through users of each partition.
        for i, user_id in enumerate(user_metadata):
            # Load and concatenate all images of a user.
            image_array, image_id_array = [], []
            labels_array, fine_grained_labels_array = [], []
            # Load and concatenate all images and labels of a user.
            for metadata in user_metadata[user_id]:
                image_id = metadata["image_id"]
                image = Image.open(
                    os.path.join(image_dir, f"{image_id}.jpg"))
                image_array.append(np.asarray(image))
                image_id_array.append(image_id)
                # Encode labels as a single string, separated by delimiter |
                labels_array.append(LABEL_DELIMITER.join(metadata["labels"]))
                fine_grained_labels_array.append(
                    LABEL_DELIMITER.join(metadata["fine_grained_labels"]))
                # Update label counter
                label_counter.update(metadata["labels"])
                fine_grained_label_counter.update(
                    metadata["fine_grained_labels"])

            partition = user_metadata[user_id][0]["partition"]
            # Multiple variable-length labels. Needs to be stored as a string.
            h5file[f'/{partition}/{user_id}/labels'] = np.asarray(
                labels_array, dtype='S')
            h5file[f'/{partition}/{user_id}/fine_grained_labels'] = np.asarray(
                fine_grained_labels_array, dtype='S')
            h5file[f'/{partition}/{user_id}/image_ids'] = np.asarray(
                image_id_array, dtype='S')
            # Tensor with dimensions [num_images,width,height,channels]
            h5file.create_dataset(
                f'/{partition}/{user_id}/images', data=np.stack(image_array))

            if (i + 1) % LOG_INTERVAL == 0:
                logger.info("Processed {0}/{1} users".format(
                    i + 1, len(user_metadata)))

        # Write metadata
        h5file['/metadata/label_counter'] = json.dumps(label_counter)
        h5file['/metadata/fine_grained_label_counter'] = json.dumps(
            fine_grained_label_counter)

    logger.info('Finished preprocess federated dataset successfully!')


def preprocess_central_dataset(image_dir: str,
                               labels_file: str,
                               output_file: str):
    """
    Process images and labels into a HDF5 (not federated) dataset where
    data is split by train/val/test partitions.

    Same parameters as `preprocess_federated_dataset`.
    """
    logger.info('Preprocessing central dataset.')
    user_metadata = load_user_metadata(labels_file)
    label_counter = Counter()
    fine_grained_label_counter = Counter()
    with h5py.File(output_file, 'w') as h5file:
        # Iterate through users of each partition.
        for i, user_id in enumerate(user_metadata):
            # Load and concatenate all images of a user.
            for metadata in user_metadata[user_id]:
                image_id = metadata["image_id"]
                image = Image.open(
                    os.path.join(image_dir, f"{image_id}.jpg"))
                partition = metadata["partition"]
                h5file.create_dataset(
                    f'/{partition}/{image_id}/image', data=np.asarray(image))
                # Encode labels as a single string, separated by delimiter |
                h5file[f'/{partition}/{image_id}/labels'] = LABEL_DELIMITER.join(
                    metadata["labels"])
                h5file[f'/{partition}/{image_id}/fine_grained_labels'] = (
                    LABEL_DELIMITER.join(metadata["fine_grained_labels"]))
                h5file[f'/{partition}/{image_id}/user_id'] = user_id
                # Update label counter
                label_counter.update(metadata["labels"])
                fine_grained_label_counter.update(
                    metadata["fine_grained_labels"])

            if (i + 1) % LOG_INTERVAL == 0:
                logger.info("Processed {0}/{1} users".format(
                    i + 1, len(user_metadata)))

        # Write metadata
        h5file['/metadata/label_counter'] = json.dumps(label_counter)
        h5file['/metadata/fine_grained_label_counter'] = json.dumps(
            fine_grained_label_counter)

    logger.info('Finished preprocessing central dataset successfully!')


if __name__ == '__main__':
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s')

    argument_parser = argparse.ArgumentParser(
        description=
        'Preprocess the images and labels of FLAIR dataset into HDF5 files.')
    argument_parser.add_argument(
        '--dataset_dir',
        required=True,
        help='Path to directory of images and label file. '
             'Can be downloaded using download_dataset.py')
    argument_parser.add_argument(
        '--output_file',
        required=True,
        help='Path to output HDF5 file that will be constructed by this script'
    )
    argument_parser.add_argument(
        '--not_group_data_by_user',
        action='store_true',
        default=False,
        help='If true, do not group data by user IDs.'
             'If false, group data by user IDs to '
             'make suitable for federated learning.')
    arguments = argument_parser.parse_args()

    image_dir = os.path.join(arguments.dataset_dir, "small_images")
    labels_file = os.path.join(arguments.dataset_dir, "labels_and_metadata.json")
    if arguments.not_group_data_by_user:
        preprocess_central_dataset(image_dir, labels_file, arguments.output_file)
    else:
        preprocess_federated_dataset(image_dir, labels_file, arguments.output_file)
