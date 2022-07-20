# -*- coding: utf-8 -*-
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.

from typing import Dict, List, Tuple
from collections import defaultdict, Counter

import os
import sys
import argparse
import json
import tensorflow as tf
import logging

logger = logging.getLogger(name=__name__)

KEY_IMAGE_BYTES = 'image/encoded_jpeg'
KEY_IMAGE_DECODED = 'image/decoded'
KEY_LABELS = 'labels'
KEY_FINE_GRAINED_LABELS = 'fine_grained_labels'
LOG_INTERVAL = 500  # Log the preprocessing progress every interval steps


def load_user_metadata_and_label_counters(
        labels_file: str) -> Tuple[Dict, Counter, Counter]:
    """
    Load labels and metadata keyed by `user_id`, and label counts.

    :param labels_file:
        A .json file with a list of labels and metadata dictionaries. Each
        dictionary has keys: `[image_id,user_id,labels,fine_grained_labels]`.
        * `image_id` is the ID of an image.
        * `user_id` is the ID of the user `image_id` belongs to.
        * `labels` is a list of 17 higher-order class labels.
        * `fine_grained_labels` is a list of 1,628 fine-grained class labels.
    :return:
        Three dictionaries. First dictionary has key being `user_id` and value
        being a list of labels and metadata for each image `user_id` owns.
        Second and third dictionaries are counts for the labels for coarse-grained
        and fine-grained taxonomies.
    """
    user_metadata = defaultdict(list)
    with open(labels_file) as f:
        metadata_list = json.load(f)

    label_counter = Counter()
    fine_grained_label_counter = Counter()
    for metadata in metadata_list:
        user_metadata[metadata["user_id"]].append(metadata)
        label_counter.update(metadata["labels"])
        fine_grained_label_counter.update(metadata["fine_grained_labels"])
    return user_metadata, label_counter, fine_grained_label_counter


def create_example(
        image_bytes: bytes,
        labels: List[int],
        fine_grained_labels: List[int]
) -> tf.train.Example:
    """Create a `tf.train.Example` for a given image and labels"""
    features = {
        KEY_IMAGE_BYTES: tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[image_bytes])),
        KEY_LABELS: tf.train.Feature(
            int64_list=tf.train.Int64List(value=labels)),
        KEY_FINE_GRAINED_LABELS: tf.train.Feature(
            int64_list=tf.train.Int64List(value=fine_grained_labels))
    }
    return tf.train.Example(features=tf.train.Features(feature=features))


def preprocess_federated_dataset(image_dir: str,
                                 labels_file: str,
                                 tfrecords_dir: str):
    """
    Process images and labels into tfrecords where data is first split by
    train/test partitions and then split again by user ID. Label to index mapping
    will be saved to `label_to_index.json` in `tfrecords_dir`.

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
    :param tfrecords_dir:
        Save directory path for tfrecords.
    """
    logger.info('Preprocessing federated tfrecords.')
    os.makedirs(tfrecords_dir, exist_ok=True)
    (user_metadata, label_counter,
     fine_grained_label_counter) = load_user_metadata_and_label_counters(labels_file)
    label_to_index = {
        label: index for index, label
        in enumerate(sorted(label_counter.keys()))}
    fine_grained_label_to_index = {
        fine_grained_label: index for index, fine_grained_label
        in enumerate(sorted(fine_grained_label_counter.keys()))}

    with open(os.path.join(tfrecords_dir, "label_to_index.json"), "w") as f:
        json.dump({
            "labels": label_to_index,
            "fine_grained_labels": fine_grained_label_to_index
        }, f, indent=4)

    for i, user_id in enumerate(user_metadata):
        partition = user_metadata[user_id][0]["partition"]

        # Load and concatenate all images and labels of a user.
        user_examples = []
        for metadata in user_metadata[user_id]:
            image_id = metadata["image_id"]
            with open(os.path.join(image_dir, f"{image_id}.jpg"),
                      'rb') as f:
                image_bytes = f.read()
            example = create_example(
                image_bytes=image_bytes,
                labels=[label_to_index[label] for label in metadata["labels"]],
                fine_grained_labels=[
                    fine_grained_label_to_index[label]
                    for label in metadata["fine_grained_labels"]
                ])
            user_examples.append(example)

        partition_dir = os.path.join(tfrecords_dir, partition)
        os.makedirs(partition_dir, exist_ok=True)
        with tf.io.TFRecordWriter(os.path.join(
                partition_dir, f'{user_id}.tfrecords')) as writer:
            for example in user_examples:
                writer.write(example.SerializeToString())

        if (i + 1) % LOG_INTERVAL == 0:
            logger.info("Processed {0}/{1} users".format(
                i + 1, len(user_metadata)))
    logger.info('Finished preprocess federated tfrecords successfully!')


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
        '--tfrecords_dir',
        required=True,
        help='Path to directory to save output tfrecords.'
    )
    arguments = argument_parser.parse_args()

    image_dir = os.path.join(arguments.dataset_dir, "small_images")
    labels_file = os.path.join(arguments.dataset_dir,
                               "labels_and_metadata.json")
    preprocess_federated_dataset(image_dir, labels_file,
                                 arguments.tfrecords_dir)
