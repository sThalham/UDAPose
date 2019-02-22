"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import keras.backend
from .dynamic import meshgrid
import tensorflow as tf


def bbox_transform_inv(boxes, deltas, mean=None, std=None):
    """ Applies deltas (usually regression results) to boxes (usually anchors).

    Before applying the deltas to the boxes, the normalization that was previously applied (in the generator) has to be removed.
    The mean and std are the mean and std as applied in the generator. They are unnormalized in this function and then applied to the boxes.

    Args
        boxes : np.array of shape (B, N, 4), where B is the batch size, N the number of boxes and 4 values for (x1, y1, x2, y2).
        deltas: np.array of same shape as boxes. These deltas (d_x1, d_y1, d_x2, d_y2) are a factor of the width/height.
        mean  : The mean value used when computing deltas (defaults to [0, 0, 0, 0]).
        std   : The standard deviation used when computing deltas (defaults to [0.2, 0.2, 0.2, 0.2]).

    Returns
        A np.array of the same shape as boxes, but with deltas applied to each box.
        The mean and std are used during training to normalize the regression values (networks love normalization).
    """
    if mean is None:
        mean = [0, 0, 0, 0]
    if std is None:
        std = [0.2, 0.2, 0.2, 0.2]

    width  = boxes[:, :, 2] - boxes[:, :, 0]
    height = boxes[:, :, 3] - boxes[:, :, 1]

    x1 = boxes[:, :, 0] + (deltas[:, :, 0] * std[0] + mean[0]) * width
    y1 = boxes[:, :, 1] + (deltas[:, :, 1] * std[1] + mean[1]) * height
    x2 = boxes[:, :, 2] + (deltas[:, :, 2] * std[2] + mean[2]) * width
    y2 = boxes[:, :, 3] + (deltas[:, :, 3] * std[3] + mean[3]) * height

    pred_boxes = keras.backend.stack([x1, y1, x2, y2], axis=2)

    return pred_boxes


def shift(shape, stride, anchors):
    """ Produce shifted anchors based on shape of the map and stride size.

    Args
        shape  : Shape to shift the anchors over.
        stride : Stride to shift the anchors with over the shape.
        anchors: The anchors to apply at each location.
    """
    shift_x = (keras.backend.arange(0, shape[1], dtype=keras.backend.floatx()) + keras.backend.constant(0.5, dtype=keras.backend.floatx())) * stride
    shift_y = (keras.backend.arange(0, shape[0], dtype=keras.backend.floatx()) + keras.backend.constant(0.5, dtype=keras.backend.floatx())) * stride

    shift_x, shift_y = meshgrid(shift_x, shift_y)
    shift_x = keras.backend.reshape(shift_x, [-1])
    shift_y = keras.backend.reshape(shift_y, [-1])

    shifts = keras.backend.stack([
        shift_x,
        shift_y,
        shift_x,
        shift_y
    ], axis=0)

    shifts            = keras.backend.transpose(shifts)
    number_of_anchors = keras.backend.shape(anchors)[0]

    k = keras.backend.shape(shifts)[0]  # number of base points = feat_h * feat_w

    shifted_anchors = keras.backend.reshape(anchors, [1, number_of_anchors, 4]) + keras.backend.cast(keras.backend.reshape(shifts, [k, 1, 4]), keras.backend.floatx())
    shifted_anchors = keras.backend.reshape(shifted_anchors, [k * number_of_anchors, 4])

    return shifted_anchors


def translation_transform_inv(boxes, deltas, deltas_pose, mean=None, std=None):
    if mean is None:
        mean = [0.0, 0.0]
    if std is None:
        std = [0.4, 0.4]

    #subTensors = []
    #for i in range(0, keras.backend.int_shape(deltas_pose)[2]):
    #    width  = deltas[:, :, 2] - deltas[:, :, 0]
    #    height = deltas[:, :, 3] - deltas[:, :, 1]

    #    x = deltas[:, :, 0] + (deltas_pose[:, :, i, 0] * std[0] + mean[0]) * width
    #    y = deltas[:, :, 1] + (deltas_pose[:, :, i, 1] * std[1] + mean[1]) * height

    #    pred_pose = keras.backend.stack([x, y], axis=2)
    #    pred_pose = keras.backend.expand_dims(pred_pose, axis=2)
    #    subTensors.append(pred_pose)
    #xy_cls = keras.backend.concatenate(subTensors, axis=2)

    num_classes = keras.backend.int_shape(deltas_pose)[2]

    #allTargets = np.repeat(targets[:, np.newaxis, :], num_classes, axis=1)
    deltas_exp = keras.backend.expand_dims(deltas, axis=2)
    deltas_exp = keras.backend.repeat_elements(deltas_exp, num_classes, axis=2)

    width = deltas_exp[:, :, :, 2] - deltas_exp[:, :, :, 0]
    height = deltas_exp[:, :, :, 3] - deltas_exp[:, :, :, 1]

    x = deltas_exp[:, :, :, 0] + (deltas_pose[:, :, :, 0] * std[0] + mean[0]) * width
    y = deltas_exp[:, :, :, 1] + (deltas_pose[:, :, :, 1] * std[1] + mean[1]) * height

    pred_pose = keras.backend.stack([x, y], axis=3)

    return pred_pose


def depth_transform_inv(boxes, deltas, deltas_pose, mean=None, std=None):
    if mean is None:
        mean = [0.0, 0.0]
    if std is None:
        std = [0.4, 0.4]

    #subTensors = []
    #for i in range(0, keras.backend.int_shape(deltas)[2]):
    #    z = deltas[:, :, i, 0] * std[0] + mean[0]
    #    pred_pose = keras.backend.stack([z], axis=2)
    #    pred_pose = keras.backend.expand_dims(pred_pose, axis=2)
    #    subTensors.append(pred_pose)
    #dep_cls = keras.backend.concatenate(subTensors, axis=2)

    #z = deltas[:, :, :, 0] * std[0] + mean[0]

    widths = deltas[:, :, 2] - deltas[:, :, 0]
    heights = deltas[:, :, 3] - deltas[:, :, 1]
    box_diag = keras.backend.sqrt((keras.backend.square(widths) + keras.backend.square(heights)))

    #n_anchors = keras.backend.int_shape(box_diag)
    #print(n_anchors)
    #scaling = keras.backend.expand_dims(box_diag, axis=0)
    #scaling = keras.backend.repeat_elements(scaling, n_anchors, axis=0)
    #print(scaling)

    #targets_dz = gt_poses[:, 2] * 100.0 / box_diag
    num_classes = keras.backend.int_shape(deltas_pose)[2]
    deltas_exp = keras.backend.expand_dims(box_diag, axis=2)
    deltas_exp = keras.backend.repeat_elements(deltas_exp, num_classes, axis=2)
    deltas_exp = keras.backend.expand_dims(deltas_exp, axis=3)
    meters = (deltas_pose[:, :, :, 0] * deltas_exp[:, :, :, 0]) * 0.01 * std[0] + mean[0]

    dep_cls = keras.backend.stack([meters], axis=3)

    return dep_cls


def rotation_transform_inv(poses, deltas, mean=None, std=None):
    if mean is None:
        mean = [0.0, 0.0, 0.0, 0.0]
    if std is None:
        std = [1.0, 1.0, 1.0, 1.0]

    subTensors = []
    for i in range(0, keras.backend.int_shape(deltas)[2]):
        rx = deltas[:, :, i, 0] * std[0] + mean[0]
        ry = deltas[:, :, i, 1] * std[1] + mean[1]
        rz = deltas[:, :, i, 2] * std[2] + mean[2]
        rw = deltas[:, :, i, 3] * std[3] + mean[3]

        pred_pose = keras.backend.stack([rx, ry, rz, rw], axis=2)
        pred_pose = keras.backend.expand_dims(pred_pose, axis=2)
        subTensors.append(pred_pose)
    pose_cls = keras.backend.concatenate(subTensors, axis=2)

    return pose_cls
