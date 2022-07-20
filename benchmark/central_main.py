# -*- coding: utf-8 -*-
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.

import atexit
import os
import tensorflow as tf
from absl import app
from absl import flags
from absl import logging

from . import flair_data, flair_metrics, flair_model

# Central training hyperparameters
flags.DEFINE_float('learning_rate', 1e-4, 'Learning rate')
flags.DEFINE_float('clipnorm', 10.0, 'Max L2 norm for gradient of each weight.')
flags.DEFINE_integer('train_batch_size', 512, 'Batch size on the clients.')
# Training loop configuration
flags.DEFINE_integer('num_epochs', 100, 'Number of total training rounds.')
flags.DEFINE_integer('eval_batch_size', 512,
                     'Batch size when evaluating on central datasets.')
# Model configuration
flags.DEFINE_string('restore_model_path', None, 'Path to pretrained model.')
flags.DEFINE_string(
    'save_model_dir', './', 'Path to directory for saving model.')
# Data configuration
flags.DEFINE_string('tfrecords_dir', None, 'Path to FLAIR tfrecords.')
flags.DEFINE_integer('image_height', 224, 'Height of input image.')
flags.DEFINE_integer('image_width', 224, 'Width of input image.')
flags.DEFINE_boolean('use_fine_grained_labels', False,
                     'use_fine_grained_labels.')
flags.DEFINE_string(
    'binary_label', None,
    'If set, train a binary classification model on the provided binary label.')

FLAGS = flags.FLAGS


def main(argv):
    if len(argv) > 1:
        raise app.UsageError('Expected no command-line arguments, '
                             'got: {}'.format(argv))

    image_shape = (256, 256, 3)
    label_to_index = flair_data.load_label_to_index(
        os.path.join(FLAGS.tfrecords_dir, "label_to_index.json"),
        FLAGS.use_fine_grained_labels)
    num_labels = len(label_to_index)

    binary_label_index = None
    if FLAGS.binary_label is not None:
        binary_label_index = label_to_index[FLAGS.binary_label]

    train_fed_data, val_fed_data, test_fed_data = flair_data.load_tfrecords_data(
        FLAGS.tfrecords_dir,
        image_shape=image_shape,
        num_labels=num_labels,
        use_fine_grained_labels=FLAGS.use_fine_grained_labels,
        binary_label_index=binary_label_index)

    if binary_label_index is not None:
        num_labels = 1

    logging.info(
        "{} training users, {} validating users".format(
            len(train_fed_data.client_ids), len(val_fed_data.client_ids)))

    def preprocess_fn(data: tf.data.Dataset,
                      is_training: bool) -> tf.data.Dataset:
        """Preprocesses `tf.data.Dataset` by shuffling and batching."""
        if is_training:
            return data.shuffle(10000).batch(FLAGS.train_batch_size)
        else:
            return data.batch(FLAGS.eval_batch_size)

    train_data = preprocess_fn(
        train_fed_data.create_tf_dataset_from_all_clients(), is_training=True)
    val_data = preprocess_fn(
        val_fed_data.create_tf_dataset_from_all_clients(), is_training=False)
    test_data = preprocess_fn(
        test_fed_data.create_tf_dataset_from_all_clients(), is_training=False)

    strategy = tf.distribute.MirroredStrategy()
    # To prevent OSError: [Errno 9] Bad file descriptor
    # https://github.com/tensorflow/tensorflow/issues/50487
    atexit.register(strategy._extended._collective_ops._pool.close)
    logging.info('Number of devices: {}'.format(strategy.num_replicas_in_sync))

    # Open a strategy scope.
    with strategy.scope():
        model = flair_model.resnet18(
            input_shape=image_shape,
            num_classes=num_labels,
            pretrained=FLAGS.restore_model_path is not None)
        if FLAGS.restore_model_path is not None:
            logging.info("Loading pretrained weights from {}".format(
                FLAGS.restore_model_path))
            model.load_weights(
                FLAGS.restore_model_path, skip_mismatch=True, by_name=True)
        model.compile(
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
            metrics=flair_metrics.metrics_builder(num_labels),
            optimizer=tf.keras.optimizers.Adam(
                FLAGS.learning_rate, clipnorm=FLAGS.clipnorm))

    os.makedirs(FLAGS.save_model_dir, exist_ok=True)
    save_model_path = os.path.join(
            FLAGS.save_model_dir, f"resnet18_central_{num_labels}labels.h5")
    model_ckpt_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=save_model_path,
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=True)

    logging.info('Training model:')
    logging.info(model.summary())

    model.fit(
        train_data,
        epochs=FLAGS.num_epochs,
        batch_size=FLAGS.train_batch_size,
        validation_data=val_data,
        validation_batch_size=FLAGS.eval_batch_size,
        callbacks=[model_ckpt_callback])

    model.load_weights(save_model_path, by_name=True)
    # final dev evaluation
    logging.info("Evaluating best model on val set.")
    val_metrics = model.evaluate(
        val_data, batch_size=FLAGS.eval_batch_size, return_dict=True)
    val_metrics = {'final val ' + k: v for k, v in
                   flair_metrics.flatten_metrics(val_metrics).items()}
    flair_metrics.print_metrics(val_metrics)

    # final test evaluation
    logging.info("Evaluating best model on test set.")
    test_metrics = model.evaluate(
        test_data, batch_size=FLAGS.eval_batch_size, return_dict=True)
    test_metrics = {'final test ' + k: v for k, v in
                    flair_metrics.flatten_metrics(test_metrics).items()}
    flair_metrics.print_metrics(test_metrics)


if __name__ == '__main__':
    app.run(main)
