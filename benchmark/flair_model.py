# -*- coding: utf-8 -*-
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2022 Apple Inc. All Rights Reserved.

import tensorflow as tf
from typing import Optional, Callable, List, Tuple

from keras.layers import (
    RandomCrop, RandomFlip, Normalization, Rescaling,
    Conv2D, ZeroPadding2D, ReLU, MaxPooling2D, BatchNormalization)
import tensorflow_addons.layers.normalizations as tfa_norms


# ImageNet statistics from https://pytorch.org/vision/stable/models.html
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_VARIANCE = [0.229 ** 2, 0.224 ** 2, 0.225 ** 2]


class FrozenBatchNormalization(BatchNormalization):
    """
    BatchNormalization layer that freezes the moving mean and average updates.
    It is intended to be used in fine-tuning a pretrained model in federated
    learning setting, where the moving mean and average will be assigned to
    the ones in the pretrained model. Only beta and gamma are updated.
    """
    def call(self, inputs, training=None):
        inputs_dtype = inputs.dtype.base_dtype
        if inputs_dtype in (tf.float16, tf.bfloat16):
            inputs = tf.cast(inputs, tf.float32)

        # Compute the axes along which to reduce the mean / variance
        input_shape = inputs.shape
        ndims = len(input_shape)
        reduction_axes = [i for i in range(ndims) if i not in self.axis]

        # Broadcasting only necessary for single-axis batch norm where the axis
        # is not the last dimension
        broadcast_shape = [1] * ndims
        broadcast_shape[self.axis[0]] = input_shape.dims[self.axis[0]].value

        def _broadcast(v):
            if (v is not None and len(v.shape) != ndims and
                    reduction_axes != list(range(ndims - 1))):
                return tf.reshape(v, broadcast_shape)
            return v

        scale, offset = _broadcast(self.gamma), _broadcast(self.beta)
        # use pretrained moving_mean and moving_variance for normalization
        mean, variance = self.moving_mean, self.moving_variance
        mean = tf.cast(mean, inputs.dtype)
        variance = tf.cast(variance, inputs.dtype)
        if offset is not None:
            offset = tf.cast(offset, inputs.dtype)
        if scale is not None:
            scale = tf.cast(scale, inputs.dtype)
        outputs = tf.nn.batch_normalization(inputs, _broadcast(mean),
                                            _broadcast(variance), offset, scale,
                                            self.epsilon)
        if inputs_dtype in (tf.float16, tf.bfloat16):
            outputs = tf.cast(outputs, inputs_dtype)

        # If some components of the shape got lost due to adjustments, fix that.
        outputs.set_shape(input_shape)
        return outputs


def conv3x3(x: tf.Tensor, scope: str, out_planes: int, stride: int = 1,
            groups: int = 1, dilation: int = 1, seed: int = 0):
    """3x3 convolution with padding"""
    x = ZeroPadding2D(padding=(dilation, dilation), name=f"{scope}_padding")(x)
    return Conv2D(
        out_planes,
        kernel_size=3,
        strides=stride,
        groups=groups,
        use_bias=False,
        dilation_rate=dilation,
        name=f"{scope}_3x3",
        kernel_initializer=tf.keras.initializers.HeNormal(seed=seed),
    )(x)


def conv1x1(x: tf.Tensor, scope: str, out_planes: int, stride: int = 1,
            seed: int = 0):
    """1x1 convolution"""
    return Conv2D(
        out_planes,
        kernel_size=1,
        strides=stride,
        use_bias=False,
        name=f"{scope}_1x1",
        kernel_initializer=tf.keras.initializers.HeNormal(seed=seed),
    )(x)


def norm(x: tf.Tensor, scope: str, use_batch_norm: bool):
    """Normalization layer"""
    if use_batch_norm:
        return FrozenBatchNormalization(axis=3, epsilon=1e-5, name=scope)(x)
    else:
        return tfa_norms.GroupNormalization(epsilon=1e-5, name=scope)(x)


def relu(x: tf.Tensor, scope: str):
    """ReLU activation layer"""
    return ReLU(name=scope)(x)


def basic_block(x: tf.Tensor, scope: str, out_planes: int, use_batch_norm: bool,
                stride: int = 1, downsample: Optional[Callable] = None,
                seed: int = 0):
    """Basic ResNet block"""
    out = conv3x3(x, f"{scope}_conv1", out_planes, stride, seed=seed)
    out = norm(out, scope=f"{scope}_norm1", use_batch_norm=use_batch_norm)
    out = relu(out, f"{scope}_relu1")
    out = conv3x3(out, f"{scope}_conv2", out_planes, seed=seed)
    out = norm(out, scope=f"{scope}_norm2", use_batch_norm=use_batch_norm)
    if downsample is not None:
        x = downsample(x)
    out += x
    out = relu(out, f"{scope}_relu2")
    return out


def block_layers(
    x: tf.Tensor,
    scope: str,
    in_planes: int,
    out_planes: int,
    blocks: int,
    use_batch_norm: bool,
    stride: int = 1,
    seed: int = 0,
):
    """Layers of ResNet block"""
    downsample = None
    if stride != 1 or in_planes != out_planes:
        # Downsample is performed when stride > 1 according to Section 3.3 in
        # https://arxiv.org/pdf/1512.03385.pdf
        def downsample(h: tf.Tensor):
            h = conv1x1(h, f"{scope}_downsample_conv", out_planes, stride)
            return norm(h, f"{scope}_downsample_norm", use_batch_norm)

    x = basic_block(x, f"{scope}_block1", out_planes, use_batch_norm, stride,
                    downsample, seed=seed)
    for i in range(1, blocks):
        x = basic_block(x, f"{scope}_block{i + 1}", out_planes, use_batch_norm,
                        seed=seed)
    return x


def create_resnet(input_shape: Tuple[int, int, int],
                  num_classes: int,
                  use_batch_norm: bool,
                  repetitions: List[int] = None,
                  initial_filters: int = 64,
                  seed: int = 0):
    """
    Creates a ResNet Keras model. Implementation follows torchvision in
    https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
    """
    img_input = tf.keras.layers.Input(shape=input_shape)

    # preprocessing layer
    x = RandomCrop(height=224, width=224)(img_input)
    x = RandomFlip()(x)
    x = Rescaling(scale=1 / 255.0)(x)
    x = Normalization(
        axis=-1, mean=IMAGENET_MEAN, variance=IMAGENET_VARIANCE)(x)

    # initial conv layer
    x = ZeroPadding2D((3, 3), name="initial_padding")(x)
    x = Conv2D(
        initial_filters,
        kernel_size=7, strides=2, use_bias=False, name="initial_conv",
        kernel_initializer=tf.keras.initializers.HeNormal(seed=seed))(x)
    x = norm(x, scope="initial_norm", use_batch_norm=use_batch_norm)
    x = relu(x, scope="initial_relu")
    x = ZeroPadding2D((1, 1), name="pooling_padding")(x)
    x = MaxPooling2D(pool_size=3, strides=2, name="initial_pooling")(x)

    # residual blocks
    x = block_layers(x, "layer1", initial_filters, 64, repetitions[0],
                     use_batch_norm, seed=seed)
    x = block_layers(x, "layer2", initial_filters, 128, repetitions[1],
                     use_batch_norm, 2, seed=seed)
    x = block_layers(x, "layer3", initial_filters, 256, repetitions[2],
                     use_batch_norm, 2, seed=seed)
    x = block_layers(x, "layer4", initial_filters, 512, repetitions[3],
                     use_batch_norm, 2, seed=seed)

    # classification layers
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_pooling")(x)
    x = tf.keras.layers.Dense(
        num_classes, name="classifier",
        kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed))(x)
    model = tf.keras.models.Model(img_input, x)
    return model


def resnet18(input_shape: Tuple[int, int, int],
             num_classes: int,
             pretrained: bool,
             seed: int = 0):
    """
    Creates a ResNet18 keras model.

    :param input_shape:
        Input image shape in [height, weight, channels.]
    :param num_classes:
        Number of output classes.
    :param pretrained:
        Whether the model is pretrained on ImageNet. If true, model will use
        BatchNormalization. If false, model will use GroupNormalization in order
        to train with differential privacy.
    :param seed:
        Random seed for initialize the weights.

    :return:
        A ResNet18 keras model
    """
    model = create_resnet(
        input_shape,
        num_classes,
        use_batch_norm=pretrained,
        repetitions=[2, 2, 2, 2],
        seed=seed)
    return model
