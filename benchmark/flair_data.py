# -*- coding: utf-8 -*-
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.

import functools
import json
import os
from typing import Dict, Tuple, Optional

import tensorflow as tf
import tensorflow_federated as tff


KEY_IMAGE_BYTES = 'image/encoded_jpeg'
KEY_IMAGE_DECODED = 'image/decoded'
KEY_LABELS = 'labels'
KEY_FINE_GRAINED_LABELS = 'fine_grained_labels'


def load_tfrecords(
        filename: str,
        image_shape: Tuple[int, int, int],
        num_labels: int,
        use_fine_grained_labels: bool,
        binary_label_index: Optional[int],
) -> tf.data.Dataset:
    """Load tfrecords from `filename` and return a `tf.data.Dataset`"""
    dataset = tf.data.TFRecordDataset([filename])
    key_labels = KEY_FINE_GRAINED_LABELS if use_fine_grained_labels else KEY_LABELS

    def parse(example_proto):
        """Parse an example to image and label in tensorflow tensor format."""
        feature_description = {
            KEY_IMAGE_BYTES: tf.io.FixedLenFeature([], tf.string),
            key_labels: tf.io.VarLenFeature(tf.int64),
        }
        example = tf.io.parse_single_example(example_proto, feature_description)
        image = tf.io.decode_jpeg(example[KEY_IMAGE_BYTES])
        labels = tf.reduce_sum(
            tf.one_hot(
                example[key_labels].values, depth=num_labels, dtype=tf.int32),
            axis=0)
        if binary_label_index is not None:
            labels = labels[binary_label_index]
        return tf.reshape(image, image_shape), labels

    return dataset.map(parse, tf.data.AUTOTUNE)


def load_label_to_index(label_to_index_file: str,
                        use_fine_grained_labels: bool) -> Dict[str, int]:
    """
    Load label to index mapping.

    :param label_to_index_file:
        Path to json file that has the label to index mapping.
    :param use_fine_grained_labels:
        Whether to load mapping for fine-grained labels.

    :return:
        A dictionary that maps label to index.
    """
    with open(label_to_index_file) as f:
        return json.load(f)[
            "fine_grained_labels" if use_fine_grained_labels else "labels"]


def load_tfrecords_data(
        tfrecords_dir: str,
        image_shape: Tuple[int, int, int],
        num_labels: int,
        use_fine_grained_labels: bool,
        binary_label_index: Optional[int] = None
) -> Tuple[tff.simulation.datasets.FilePerUserClientData, ...]:
    """
    Load tfrecords data into TFF format.

    :param tfrecords_dir:
        Directory with all tfrecords saved, processed by `prepapre_tfrecords.py`.
    :param image_shape:
        3D tuple indicating shape of image [height, weight, channels].
    :param num_labels:
        Number of labels.
    :param use_fine_grained_labels:
        Whether to use fine-grained labels.
    :param binary_label_index:
        Optional integer. If set, label will be a binary value for the given
        `binary_label_index`, and other label indices will be ignored.

    :return:
        A tuple of three `tff.simulation.datasets.FilePerUserClientData` object
        for train, val and test set respectively.
    """
    def get_client_ids_to_files(partition: str):
        """Get the tfrecords filenames for a train/val/test partition"""
        partition_dir = os.path.join(tfrecords_dir, partition)
        partition_client_files = os.listdir(partition_dir)
        return {
            client_file.split(".tfrecords")[0]: os.path.join(
                partition_dir, client_file)
            for client_file in partition_client_files
        }

    return tuple([
        tff.simulation.datasets.FilePerUserClientData(
            client_ids_to_files=get_client_ids_to_files(partition),
            dataset_fn=functools.partial(
                load_tfrecords,
                image_shape=image_shape,
                num_labels=num_labels,
                use_fine_grained_labels=use_fine_grained_labels,
                binary_label_index=binary_label_index)
        )
        for partition in ['train', 'val', 'test']
    ])
