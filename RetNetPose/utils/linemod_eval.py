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

#from pycocotools.cocoeval import COCOeval

import keras
import numpy as np
import json
import pyquaternion
import math
import transforms3d as tf3d
import geometry
import os
import cv2
from .pose_error import add
from .pose_error import adi
from .pose_error import reproj
from .pose_error import re
from .pose_error import te

import progressbar
assert(callable(progressbar.progressbar)), "Using wrong progressbar module, install 'progressbar2' instead."

# LineMOD
fxkin = 572.41140
fykin = 573.57043
cxkin = 325.26110
cykin = 242.04899


threeD_boxes = np.ndarray((15, 8, 3), dtype=np.float32)
threeD_boxes[0, :, :] = np.array([[0.038, 0.039, 0.046],  # ape [76, 78, 92]
                                     [0.038, 0.039, -0.046],
                                     [0.038, -0.039, -0.046],
                                     [0.038, -0.039, 0.046],
                                     [-0.038, 0.039, 0.046],
                                     [-0.038, 0.039, -0.046],
                                     [-0.038, -0.039, -0.046],
                                     [-0.038, -0.039, 0.046]])
threeD_boxes[1, :, :] = np.array([[0.108, 0.061, 0.1095],  # benchvise [216, 122, 219]
                                     [0.108, 0.061, -0.1095],
                                     [0.108, -0.061, -0.1095],
                                     [0.108, -0.061, 0.1095],
                                     [-0.108, 0.061, 0.1095],
                                     [-0.108, 0.061, -0.1095],
                                     [-0.108, -0.061, -0.1095],
                                     [-0.108, -0.061, 0.1095]])
threeD_boxes[2, :, :] = np.array([[0.083, 0.0825, 0.037],  # bowl [166, 165, 74]
                                     [0.083, 0.0825, -0.037],
                                     [0.083, -0.0825, -0.037],
                                     [0.083, -0.0825, 0.037],
                                     [-0.083, 0.0825, 0.037],
                                     [-0.083, 0.0825, -0.037],
                                     [-0.083, -0.0825, -0.037],
                                     [-0.083, -0.0825, 0.037]])
threeD_boxes[3, :, :] = np.array([[0.0685, 0.0715, 0.05],  # camera [137, 143, 100]
                                     [0.0685, 0.0715, -0.05],
                                     [0.0685, -0.0715, -0.05],
                                     [0.0685, -0.0715, 0.05],
                                     [-0.0685, 0.0715, 0.05],
                                     [-0.0685, 0.0715, -0.05],
                                     [-0.0685, -0.0715, -0.05],
                                     [-0.0685, -0.0715, 0.05]])
threeD_boxes[4, :, :] = np.array([[0.0505, 0.091, 0.097],  # can [101, 182, 194]
                                     [0.0505, 0.091, -0.097],
                                     [0.0505, -0.091, -0.097],
                                     [0.0505, -0.091, 0.097],
                                     [-0.0505, 0.091, 0.097],
                                     [-0.0505, 0.091, -0.097],
                                     [-0.0505, -0.091, -0.097],
                                     [-0.0505, -0.091, 0.097]])
threeD_boxes[5, :, :] = np.array([[0.0335, 0.064, 0.0585],  # cat [67, 128, 117]
                                     [0.0335, 0.064, -0.0585],
                                     [0.0335, -0.064, -0.0585],
                                     [0.0335, -0.064, 0.0585],
                                     [-0.0335, 0.064, 0.0585],
                                     [-0.0335, 0.064, -0.0585],
                                     [-0.0335, -0.064, -0.0585],
                                     [-0.0335, -0.064, 0.0585]])
threeD_boxes[6, :, :] = np.array([[0.059, 0.046, 0.0475],  # mug [118, 92, 95]
                                     [0.059, 0.046, -0.0475],
                                     [0.059, -0.046, -0.0475],
                                     [0.059, -0.046, 0.0475],
                                     [-0.059, 0.046, 0.0475],
                                     [-0.059, 0.046, -0.0475],
                                     [-0.059, -0.046, -0.0475],
                                     [-0.059, -0.046, 0.0475]])
threeD_boxes[7, :, :] = np.array([[0.115, 0.038, 0.104],  # drill [230, 76, 208]
                                     [0.115, 0.038, -0.104],
                                     [0.115, -0.038, -0.104],
                                     [0.115, -0.038, 0.104],
                                     [-0.115, 0.038, 0.104],
                                     [-0.115, 0.038, -0.104],
                                     [-0.115, -0.038, -0.104],
                                     [-0.115, -0.038, 0.104]])
threeD_boxes[8, :, :] = np.array([[0.052, 0.0385, 0.043],  # duck [104, 77, 86]
                                     [0.052, 0.0385, -0.043],
                                     [0.052, -0.0385, -0.043],
                                     [0.052, -0.0385, 0.043],
                                     [-0.052, 0.0385, 0.043],
                                     [-0.052, 0.0385, -0.043],
                                     [-0.052, -0.0385, -0.043],
                                     [-0.052, -0.0385, 0.043]])
threeD_boxes[9, :, :] = np.array([[0.075, 0.0535, 0.0345],  # eggbox [150, 107, 69]
                                     [0.075, 0.0535, -0.0345],
                                     [0.075, -0.0535, -0.0345],
                                     [0.075, -0.0535, 0.0345],
                                     [-0.075, 0.0535, 0.0345],
                                     [-0.075, 0.0535, -0.0345],
                                     [-0.075, -0.0535, -0.0345],
                                     [-0.075, -0.0535, 0.0345]])
threeD_boxes[10, :, :] = np.array([[0.0185, 0.039, 0.0865],  # glue [37, 78, 173]
                                     [0.0185, 0.039, -0.0865],
                                     [0.0185, -0.039, -0.0865],
                                     [0.0185, -0.039, 0.0865],
                                     [-0.0185, 0.039, 0.0865],
                                     [-0.0185, 0.039, -0.0865],
                                     [-0.0185, -0.039, -0.0865],
                                     [-0.0185, -0.039, 0.0865]])
threeD_boxes[11, :, :] = np.array([[0.0505, 0.054, 0.04505],  # holepuncher [101, 108, 91]
                                     [0.0505, 0.054, -0.04505],
                                     [0.0505, -0.054, -0.04505],
                                     [0.0505, -0.054, 0.04505],
                                     [-0.0505, 0.054, 0.04505],
                                     [-0.0505, 0.054, -0.04505],
                                     [-0.0505, -0.054, -0.04505],
                                     [-0.0505, -0.054, 0.04505]])
threeD_boxes[12, :, :] = np.array([[0.115, 0.038, 0.104],  # drill [230, 76, 208]
                                     [0.115, 0.038, -0.104],
                                     [0.115, -0.038, -0.104],
                                     [0.115, -0.038, 0.104],
                                     [-0.115, 0.038, 0.104],
                                     [-0.115, 0.038, -0.104],
                                     [-0.115, -0.038, -0.104],
                                     [-0.115, -0.038, 0.104]])
threeD_boxes[13, :, :] = np.array([[0.129, 0.059, 0.0705],  # iron [258, 118, 141]
                                     [0.129, 0.059, -0.0705],
                                     [0.129, -0.059, -0.0705],
                                     [0.129, -0.059, 0.0705],
                                     [-0.129, 0.059, 0.0705],
                                     [-0.129, 0.059, -0.0705],
                                     [-0.129, -0.059, -0.0705],
                                     [-0.129, -0.059, 0.0705]])
threeD_boxes[14, :, :] = np.array([[0.047, 0.0735, 0.0925],  # phone [94, 147, 185]
                                     [0.047, 0.0735, -0.0925],
                                     [0.047, -0.0735, -0.0925],
                                     [0.047, -0.0735, 0.0925],
                                     [-0.047, 0.0735, 0.0925],
                                     [-0.047, 0.0735, -0.0925],
                                     [-0.047, -0.0735, -0.0925],
                                     [-0.047, -0.0735, 0.0925]])


def create_point_cloud(depth, ds):

    rows, cols = depth.shape

    depRe = depth.reshape(rows * cols)
    zP = np.multiply(depRe, ds)

    x, y = np.meshgrid(np.arange(0, cols, 1), np.arange(0, rows, 1), indexing='xy')
    yP = y.reshape(rows * cols) - cykin
    xP = x.reshape(rows * cols) - cxkin
    yP = np.multiply(yP, zP)
    xP = np.multiply(xP, zP)
    yP = np.divide(yP, fykin)
    xP = np.divide(xP, fxkin)

    cloud_final = np.transpose(np.array((xP, yP, zP)))

    return cloud_final


def boxoverlap(a, b):
    a = np.array([a[0], a[1], a[0] + a[2], a[1] + a[3]])
    b = np.array([b[0], b[1], b[0] + b[2], b[1] + b[3]])

    x1 = np.amax(np.array([a[0], b[0]]))
    y1 = np.amax(np.array([a[1], b[1]]))
    x2 = np.amin(np.array([a[2], b[2]]))
    y2 = np.amin(np.array([a[3], b[3]]))

    wid = x2-x1+1
    hei = y2-y1+1
    inter = wid * hei
    aarea = (a[2] - a[0] + 1) * (a[3] - a[1] + 1)
    barea = (b[2] - b[0] + 1) * (b[3] - b[1] + 1)
    # intersection over union overlap
    ovlap = inter / (aarea + barea - inter)
    # set invalid entries to 0 overlap
    maskwid = wid <= 0
    maskhei = hei <= 0
    np.where(ovlap, maskwid, 0)
    np.where(ovlap, maskhei, 0)

    return ovlap


def evaluate_linemod(generator, model, threshold=0.05):
    threshold = 0.5
    """ Use the pycocotools to evaluate a COCO model on a dataset.

    Args
        generator : The generator for generating the evaluation data.
        model     : The model to evaluate.
        threshold : The score threshold to use.
    """
    # start collecting results
    results = []
    image_ids = []
    image_indices = []
    idx = 0

    tp = np.zeros((16), dtype=np.uint32)
    fp = np.zeros((16), dtype=np.uint32)
    fn = np.zeros((16), dtype=np.uint32)
    xyD = []
    xyzD = []
    zD = []
    less5cm_imgplane = []
    less5cm = []
    less10cm = []
    less15cm = []
    less20cm = []
    less25cm = []
    rotD = []
    less5deg = []
    less10deg = []
    less15deg = []
    less20deg = []
    less25deg = []

    # load meshes
    mesh_path = "/home/sthalham/data/LINEMOD/models/"
    sub = os.listdir(mesh_path)
    mesh_dict = {}
    for m in sub:
        if m.endswith('.ply'):
            name = m[:-4]
            key = str(int(name[-2:]))
            mesh = np.genfromtxt(mesh_path+m, skip_header=16, usecols=(0, 1, 2))
            mask = np.where(mesh[:, 0] != 3)
            mesh = mesh[mask]
            mesh_dict[key] = mesh

    for index in progressbar.progressbar(range(generator.size()), prefix='LineMOD evaluation: '):

        image_raw = generator.load_image(index)
        image = generator.preprocess_image(image_raw)
        image, scale = generator.resize_image(image)

        if keras.backend.image_data_format() == 'channels_first':
            image = image.transpose((2, 0, 1))

        # run network
        #boxes, trans, deps, rots, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
        #boxes, trans, deps, rol, pit, ya, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
        boxes, boxes3D, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))

        # correct boxes for image scale
        boxes /= scale

        # change to (x, y, w, h) (MS COCO standard)
        boxes[:, :, 2] -= boxes[:, :, 0]
        boxes[:, :, 3] -= boxes[:, :, 1]

        # target annotation
        anno = generator.load_annotations(index)
        if len(anno['labels']) > 1:
            continue
        else:
            t_cat = int(anno['labels']) + 1
        t_bbox = np.asarray(anno['bboxes'], dtype=np.float32)[0]
        t_tra = anno['poses'][0][:3]
        t_rot = anno['poses'][0][3:]
        fn[t_cat] += 1
        fnit = True

        # compute predicted labels and scores
        for box, box3D, score, label in zip(boxes[0], boxes3D[0], scores[0], labels[0]):
            # scores are sorted, so we can break
            if score < threshold:
                continue

            if label < 0:
                continue
            cls = generator.label_to_inv_label(label)
            control_points = box3D[(cls - 1), :]

            # append detection for each positively labeled class
            image_result = {
                'image_id'    : generator.image_ids[index],
                'category_id' : generator.label_to_inv_label(label),
                'score'       : float(score),
                'bbox'        : box.tolist(),
                'pose'        : control_points.tolist()
            }

            # append detection to results
            results.append(image_result)

            if cls == t_cat:
                b1 = np.array([box[0], box[1], box[0] + box[2], box[1] + box[3]])
                b2 = np.array([t_bbox[0], t_bbox[1], t_bbox[2], t_bbox[3]])
                IoU = boxoverlap(b1, b2)
                # occurences of 2 or more instances not possible in LINEMOD
                if IoU > 0.5:
                    if fnit is True:
                        tp[t_cat] += 1
                        fn[t_cat] -= 1
                        fnit = False

                        path = generator.load_image_dep(index)
                        image_dep = cv2.imread(path, cv2.IMREAD_UNCHANGED)

                        dep_val = image_dep[int(box[1] + (box[3] * 0.5)), int(box[0] + (box[2] * 0.5))]*0.001
                        if cls == 1:
                            dep = dep_val + 0.041
                        elif cls == 2:
                            dep = dep_val + 0.0928
                        elif cls == 3:
                            dep = dep_val + 0.0675
                        elif cls == 4:
                            dep = dep_val + 0.0633
                        elif cls == 5:
                            dep = dep_val + 0.0795
                        elif cls == 6:
                            dep = dep_val + 0.052
                        elif cls == 7:
                            dep = dep_val + 0.0508
                        elif cls == 8:
                            dep = dep_val + 0.0853
                        elif cls == 9:
                            dep = dep_val + 0.0445
                        elif cls == 10:
                            dep = dep_val + 0.0543
                        elif cls == 11:
                            dep = dep_val + 0.048
                        elif cls == 12:
                            dep = dep_val + 0.05
                        elif cls == 13:
                            dep = dep_val + 0.0862
                        elif cls == 14:
                            dep = dep_val + 0.0888
                        elif cls == 15:
                            dep = dep_val + 0.071

                        x_o = (((box[0] + (box[2] * 0.5)) - cxkin) * dep) / fxkin
                        y_o = (((box[1] + (box[3] * 0.5)) - cykin) * dep) / fykin

                        itvec = np.array([x_o, y_o, dep], dtype=np.float32)

                        obj_points = np.ascontiguousarray(threeD_boxes[cls-1, :, :], dtype=np.float32) #.reshape((8, 1, 3))
                        est_points = np.ascontiguousarray(control_points.T, dtype=np.float32).reshape((8, 1, 2))

                        K = np.float32([fxkin, 0., cxkin, 0., fykin, cykin, 0., 0., 1.]).reshape(3, 3)

                        #retval, orvec, otvec = cv2.solvePnP(obj_points, est_points, K, None, None, None, False, cv2.SOLVEPNP_ITERATIVE)
                        retval, orvec, otvec, inliers = cv2.solvePnPRansac(objectPoints=obj_points,
                                                                           imagePoints=est_points, cameraMatrix=K,
                                                                           distCoeffs=None, rvec=None, tvec=itvec,
                                                                           useExtrinsicGuess=True, iterationsCount=100,
                                                                           reprojectionError=5.0, confidence=0.99,
                                                                           flags=cv2.SOLVEPNP_ITERATIVE)

                        rmat, _ = cv2.Rodrigues(orvec)
                        t_rmat, _ = cv2.Rodrigues(t_rot)
                        t_rmat = tf3d.euler.euler2mat(t_rot[0], t_rot[1], t_rot[2])
                        rd = re(t_rmat, rmat)
                        #xyz = te((np.asarray(t_tra)*0.001), (otvec.T))
                        xyz = te((np.asarray(t_tra) * 0.001), (itvec.T))

                        if not math.isnan(rd):
                            rotD.append(rd)
                            if (rd) < 5.0:
                                less5deg.append(rd)
                            if (rd) < 10.0:
                                less10deg.append(rd)
                            if (rd) < 15.0:
                                less15deg.append(rd)
                            if (rd) < 20.0:
                                less20deg.append(rd)
                            if (rd) < 25.0:
                                less25deg.append(rd)

                        if not math.isnan(xyz):
                            xyzD.append(xyz)
                            if xyz < 0.05:
                                less5cm.append(xyz)
                            if xyz < 0.1:
                                less10cm.append(xyz)
                            if xyz < 0.15:
                                less15cm.append(xyz)
                            if xyz < 0.2:
                                less20cm.append(xyz)
                            if xyz < 0.25:
                                less25cm.append(xyz)

                else:
                    fp[t_cat] += 1

        # append image to list of processed images
        image_ids.append(generator.image_ids[index])
        image_indices.append(index)
        idx += 1

    if not len(results):
        return

    # write output
    json.dump(results, open('{}_bbox_results.json'.format(generator.set_name), 'w'), indent=4)
    #json.dump(image_ids, open('{}_processed_image_ids.json'.format(generator.set_name), 'w'), indent=4)

    detPre = [0] * 16
    detRec = [0] * 16

    np.set_printoptions(precision=2)
    for ind in range(1, 16):
        if ind == 0:
            continue

        if tp[ind] == 0:
            detPre[ind] = 0.0
            detRec[ind] = 0.0
        else:
            detRec[ind] = tp[ind] / (tp[ind] + fn[ind])
            detPre[ind] = tp[ind] / (tp[ind] + fp[ind])

        #print('precision category ', ind, ': ', detPre[ind])
        #print('recall category ', ind, ': ', detRec[ind])

    dataset_recall = sum(tp) / (sum(tp) + sum(fp))
    dataset_precision = sum(tp) / (sum(tp) + sum(fn))
    less5cm = len(less5cm)/len(xyzD)
    less10cm = len(less10cm) / len(xyzD)
    less15cm = len(less15cm) / len(xyzD)
    less20cm = len(less20cm) / len(xyzD)
    less25cm = len(less25cm) / len(xyzD)
    less5deg = len(less5deg) / len(rotD)
    less10deg = len(less10deg) / len(rotD)
    less15deg = len(less15deg) / len(rotD)
    less20deg = len(less20deg) / len(rotD)
    less25deg = len(less25deg) / len(rotD)
    print(' ')
    print('dataset recall: ', dataset_recall, '%')
    print('dataset precision: ', dataset_precision, '%')
    print('linemod::percent below 5 cm: ', less5cm, '%')
    print('linemod::percent below 10 cm: ', less10cm, '%')
    print('linemod::percent below 15 cm: ', less15cm, '%')
    print('linemod::percent below 20 cm: ', less20cm, '%')
    print('linemod::percent below 25 cm: ', less25cm, '%')
    print('linemod::percent below 5 deg: ', less5deg, '%')
    print('linemod::percent below 10 deg: ', less10deg, '%')
    print('linemod::percent below 15 deg: ', less15deg, '%')
    print('linemod::percent below 20 deg: ', less20deg, '%')
    print('linemod::percent below 25 deg: ', less25deg, '%')

    return dataset_recall, dataset_precision, less5cm, less5deg
