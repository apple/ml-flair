# -*- coding: utf-8 -*-
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.

import sys
import numpy as np
import tensorflow as tf

from keras.utils import metrics_utils
from typing import List, Optional, Dict


class ConfusionMatrixMetrics(tf.keras.metrics.AUC):
    """
    Base class for metrics based on confusion matrix, including precision,
    recall, F1 score and averaged precision.
    Please refer https://www.tensorflow.org/api_docs/python/tf/keras/metrics/AUC
    for arguments description.
    """
    def __init__(self,
                 num_labels: int,
                 multi_label: bool,
                 num_thresholds: int = 200,
                 name: Optional[str] = None,
                 dtype: Optional[tf.DType] = None,
                 thresholds: Optional[List[float]] = None,
                 label_weights: Optional[List[float]] = None,
                 from_logits: bool = False):

        if not multi_label:
            num_labels = None

        self._num_labels = num_labels
        if isinstance(self, (Precision, Recall, F1)):
            thresholds = metrics_utils.parse_init_thresholds(
                thresholds, default_threshold=0.5)

        super(ConfusionMatrixMetrics, self).__init__(
            num_thresholds=num_thresholds,
            curve='ROC',
            summation_method='interpolation',
            name=name,
            dtype=dtype,
            thresholds=thresholds,
            multi_label=multi_label,
            num_labels=num_labels,
            label_weights=label_weights,
            from_logits=from_logits)

    def result(self):
        raise NotImplementedError(
            "ConfusionMatrixMetrics does not return any result")

    def get_config(self):
        config = super(ConfusionMatrixMetrics, self).get_config()
        # Pop unrelated arguments
        config.pop("curve")
        config.pop("summation_method")
        # Add arguments in __init__ to pass TFF metrics builder checks
        config['thresholds'] = self.thresholds[1:-1]
        config['num_labels'] = self._num_labels
        config['from_logits'] = self._from_logits
        return config

    def update_state(self, *args, **kwargs):
        # Return None to pass TFF metrics builder checks
        super(ConfusionMatrixMetrics, self).update_state(*args, **kwargs)
        return

    def get_precision(self):
        tp, fp = self.true_positives, self.false_positives
        return tf.squeeze(tf.math.divide_no_nan(tp, tf.math.add(tp, fp)))

    def get_recall(self):
        tp, fn = self.true_positives, self.false_negatives
        return tf.squeeze(tf.math.divide_no_nan(tp, tf.math.add(tp, fn)))

    def get_macro_average(self, by_label_metrics):
        assert self.multi_label
        if self.label_weights is None:
            macro_average = tf.reduce_mean(by_label_metrics)
        else:
            macro_average = tf.reduce_sum(by_label_metrics * self.label_weights
                                          ) / tf.reduce_sum(self.label_weights)
        return macro_average


class AveragedPrecision(ConfusionMatrixMetrics):
    """
    Averaged precision metrics. Implementation follows sklearn in
    https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html
    """
    def get_averaged_precision(self):
        precision = self.get_precision()
        recall = self.get_recall()
        recall_diff = recall[1:] - recall[:-1]
        return -tf.reduce_sum(recall_diff * precision[:-1], axis=0)

    def result(self):
        averaged_precision = self.get_averaged_precision()
        if self.multi_label:
            averaged_precision = self.get_macro_average(averaged_precision)
        return averaged_precision


class Precision(ConfusionMatrixMetrics):
    """Precision metrics"""
    def result(self):
        precision = self.get_precision()[1:-1]
        if self.multi_label:
            precision = self.get_macro_average(precision)
        return precision


class Recall(ConfusionMatrixMetrics):
    """Recall metrics"""
    def result(self):
        recall = self.get_recall()[1:-1]
        if self.multi_label:
            recall = self.get_macro_average(recall)
        return recall


class F1(ConfusionMatrixMetrics):
    """F1 score metrics"""
    def result(self):
        precision = self.get_precision()[1:-1]
        recall = self.get_recall()[1:-1]
        if self.multi_label:
            precision = self.get_macro_average(precision)
            recall = self.get_macro_average(recall)
        return tf.math.divide_no_nan(
            2 * precision * recall, tf.math.add(precision, recall))


def metrics_builder(num_labels: int) -> List[tf.keras.metrics.Metric]:
    """
    Build a list of metrics to track during training.

    :param num_labels:
        Number of class labels.
    :return:
        A list of `tf.keras.metrics.Metric` object.
    """
    metrics = [
        tf.keras.metrics.BinaryCrossentropy(name="loss", from_logits=True),
    ]
    metric_classes = [Precision, Recall, F1, AveragedPrecision]
    metric_names = ["precision", "recall", "f1", "averaged_precision"]
    for metric_class, metric_name in zip(metric_classes, metric_names):
        if num_labels == 1:
            metrics.append(
                metric_class(num_labels=num_labels, multi_label=False,
                             from_logits=True, name=metric_name))
        else:
            metrics.extend(
                [
                    metric_class(num_labels=num_labels, multi_label=True,
                                 from_logits=True, name=f"macro_{metric_name}"),
                    metric_class(num_labels=num_labels, multi_label=False,
                                 from_logits=True, name=f"micro_{metric_name}")
                ])
    return metrics


def flatten_metrics(nested_metrics, prefix: Optional[str] = None
                    ) -> Dict[str, float]:
    """
    Flatten a nested metrics structure to a dictionary where key is the metric
    name and value is the metric value in float.

    :param nested_metrics:
        A nested metrics structure.
    :param prefix:
        Optional prefix description for a metrics.

    :return:
        Dictionary of metrics names and metrics values.
    """
    flattened_metrics = {}

    if isinstance(nested_metrics, dict):
        for key, value in nested_metrics.items():
            flattened_metrics.update(flatten_metrics(
                value, str(key) if prefix is None else f'{prefix} {key}'))
    elif isinstance(nested_metrics, (np.ndarray, list, tuple)):
        if len(nested_metrics) == 1:
            flattened_metrics[prefix] = float(nested_metrics[0])
        else:
            for index, value in enumerate(nested_metrics):
                flattened_metrics.update(flatten_metrics(
                    value, str(index) if prefix is None else f'{prefix} {index}'
                ))
    else:
        flattened_metrics[prefix] = float(nested_metrics)
    return flattened_metrics


def print_metrics(metrics: Dict[str, float], iteration: Optional[int] = None):
    """
    Print a dictionary of metrics names and metrics values.

    :param metrics:
        A dictionary of metrics names and metrics values.
    :param iteration:
        Optional value indicating training iteration.
    """
    if iteration is not None:
        sys.stdout.write('Metrics at iteration {}:\n'.format(iteration))
    for key, value in metrics.items():
        sys.stdout.write('    {:<50}: {}\n'.format(key, value))
