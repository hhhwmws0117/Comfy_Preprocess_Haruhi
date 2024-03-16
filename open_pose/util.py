import math
import numpy as np
import matplotlib
import cv2
from typing import List, Tuple, Union

from .body import BodyResult, Keypoint

eps = 0.01


def smart_resize(x, s):
    Ht, Wt = s
    if x.ndim == 2:
        Ho, Wo = x.shape
        Co = 1
    else:
        Ho, Wo, Co = x.shape
    if Co == 3 or Co == 1:
        k = float(Ht + Wt) / float(Ho + Wo)
        return cv2.resize(x, (int(Wt), int(Ht)), interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_LANCZOS4)
    else:
        return np.stack([smart_resize(x[:, :, i], s) for i in range(Co)], axis=2)


def smart_resize_k(x, fx, fy):
    if x.ndim == 2:
        Ho, Wo = x.shape
        Co = 1
    else:
        Ho, Wo, Co = x.shape
    Ht, Wt = Ho * fy, Wo * fx
    if Co == 3 or Co == 1:
        k = float(Ht + Wt) / float(Ho + Wo)
        return cv2.resize(x, (int(Wt), int(Ht)), interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_LANCZOS4)
    else:
        return np.stack([smart_resize_k(x[:, :, i], fx, fy) for i in range(Co)], axis=2)


def padRightDownCorner(img, stride, padValue):
    h = img.shape[0]
    w = img.shape[1]

    pad = 4 * [None]
    pad[0] = 0 # up
    pad[1] = 0 # left
    pad[2] = 0 if (h % stride == 0) else stride - (h % stride) # down
    pad[3] = 0 if (w % stride == 0) else stride - (w % stride) # right

    img_padded = img
    pad_up = np.tile(img_padded[0:1, :, :]*0 + padValue, (pad[0], 1, 1))
    img_padded = np.concatenate((pad_up, img_padded), axis=0)
    pad_left = np.tile(img_padded[:, 0:1, :]*0 + padValue, (1, pad[1], 1))
    img_padded = np.concatenate((pad_left, img_padded), axis=1)
    pad_down = np.tile(img_padded[-2:-1, :, :]*0 + padValue, (pad[2], 1, 1))
    img_padded = np.concatenate((img_padded, pad_down), axis=0)
    pad_right = np.tile(img_padded[:, -2:-1, :]*0 + padValue, (1, pad[3], 1))
    img_padded = np.concatenate((img_padded, pad_right), axis=1)

    return img_padded, pad


def transfer(model, model_weights):
    transfered_model_weights = {}
    for weights_name in model.state_dict().keys():
        transfered_model_weights[weights_name] = model_weights['.'.join(weights_name.split('.')[1:])]
    return transfered_model_weights


def draw_bodypose(canvas: np.ndarray, keypoints: List[Keypoint]) -> np.ndarray:
    """
    Draw keypoints and limbs representing body pose on a given canvas.

    Args:
        canvas (np.ndarray): A 3D numpy array representing the canvas (image) on which to draw the body pose.
        keypoints (List[Keypoint]): A list of Keypoint objects representing the body keypoints to be drawn.

    Returns:
        np.ndarray: A 3D numpy array representing the modified canvas with the drawn body pose.

    Note:
        The function expects the x and y coordinates of the keypoints to be normalized between 0 and 1.
    """
    H, W, C = canvas.shape
    stickwidth = 4

    limbSeq = [
        [2, 3], [2, 6], [3, 4], [4, 5], 
        [6, 7], [7, 8], [2, 9], [9, 10], 
        [10, 11], [2, 12], [12, 13], [13, 14], 
        [2, 1], [1, 15], [15, 17], [1, 16], 
        [16, 18],
    ]

    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

    for (k1_index, k2_index), color in zip(limbSeq, colors):
        keypoint1 = keypoints[k1_index - 1]
        keypoint2 = keypoints[k2_index - 1]

        if keypoint1 is None or keypoint2 is None:
            continue

        Y = np.array([keypoint1.x, keypoint2.x]) * float(W)
        X = np.array([keypoint1.y, keypoint2.y]) * float(H)
        mX = np.mean(X)
        mY = np.mean(Y)
        length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
        polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
        cv2.fillConvexPoly(canvas, polygon, [int(float(c) * 0.6) for c in color])

    for keypoint, color in zip(keypoints, colors):
        if keypoint is None:
            continue

        x, y = keypoint.x, keypoint.y
        x = int(x * W)
        y = int(y * H)
        cv2.circle(canvas, (int(x), int(y)), 4, color, thickness=-1)

    return canvas


def draw_handpose(canvas: np.ndarray, keypoints: Union[List[Keypoint], None]) -> np.ndarray:
    """
    Draw keypoints and connections representing hand pose on a given canvas.

    Args:
        canvas (np.ndarray): A 3D numpy array representing the canvas (image) on which to draw the hand pose.
        keypoints (List[Keypoint]| None): A list of Keypoint objects representing the hand keypoints to be drawn
                                          or None if no keypoints are present.

    Returns:
        np.ndarray: A 3D numpy array representing the modified canvas with the drawn hand pose.

    Note:
        The function expects the x and y coordinates of the keypoints to be normalized between 0 and 1.
    """
    if not keypoints:
        return canvas
    
    H, W, C = canvas.shape

    edges = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10], \
             [10, 11], [11, 12], [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]

    for ie, (e1, e2) in enumerate(edges):
        k1 = keypoints[e1]
        k2 = keypoints[e2]
        if k1 is None or k2 is None:
            continue
        
        x1 = int(k1.x * W)
        y1 = int(k1.y * H)
        x2 = int(k2.x * W)
        y2 = int(k2.y * H)
        if x1 > eps and y1 > eps and x2 > eps and y2 > eps:
            cv2.line(canvas, (x1, y1), (x2, y2), matplotlib.colors.hsv_to_rgb([ie / float(len(edges)), 1.0, 1.0]) * 255, thickness=2)

    for keypoint in keypoints:
        x, y = keypoint.x, keypoint.y
        x = int(x * W)
        y = int(y * H)
        if x > eps and y > eps:
            cv2.circle(canvas, (x, y), 4, (0, 0, 255), thickness=-1)
    return canvas


def draw_facepose(canvas: np.ndarray, keypoints: Union[List[Keypoint], None]) -> np.ndarray:
    """
    Draw keypoints representing face pose on a given canvas.

    Args:
        canvas (np.ndarray): A 3D numpy array representing the canvas (image) on which to draw the face pose.
        keypoints (List[Keypoint]| None): A list of Keypoint objects representing the face keypoints to be drawn
                                          or None if no keypoints are present.

    Returns:
        np.ndarray: A 3D numpy array representing the modified canvas with the drawn face pose.

    Note:
        The function expects the x and y coordinates of the keypoints to be normalized between 0 and 1.
    """    
    if not keypoints:
        return canvas
    
    H, W, C = canvas.shape
    for keypoint in keypoints:
        x, y = keypoint.x, keypoint.y
        x = int(x * W)
        y = int(y * H)
        if x > eps and y > eps:
            cv2.circle(canvas, (x, y), 3, (255, 255, 255), thickness=-1)
    return canvas


# detect hand according to body pose keypoints
# please refer to https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/src/openpose/hand/handDetector.cpp
def handDetect(body: BodyResult, oriImg) -> List[Tuple[int, int, int, bool]]:
    """
    Detect hands in the input body pose keypoints and calculate the bounding box for each hand.

    Args:
        body (BodyResult): A BodyResult object containing the detected body pose keypoints.
        oriImg (numpy.ndarray): A 3D numpy array representing the original input image.

    Returns:
        List[Tuple[int, int, int, bool]]: A list of tuples, each containing the coordinates (x, y) of the top-left
                                          corner of the bounding box, the width (height) of the bounding box, and
                                          a boolean flag indicating whether the hand is a left hand (True) or a
                                          right hand (False).

    Notes:
        - The width and height of the bounding boxes are equal since the network requires squared input.
        - The minimum bounding box size is 20 pixels.
    """
    ratioWristElbow = 0.33
    detect_result = []
    image_height, image_width = oriImg.shape[0:2]
    
    keypoints = body.keypoints
    # right hand: wrist 4, elbow 3, shoulder 2
    # left hand: wrist 7, elbow 6, shoulder 5
    left_shoulder = keypoints[5]
    left_elbow = keypoints[6]
    left_wrist = keypoints[7]
    right_shoulder = keypoints[2]
    right_elbow = keypoints[3]
    right_wrist = keypoints[4]

    # if any of three not detected
    has_left = all(keypoint is not None for keypoint in (left_shoulder, left_elbow, left_wrist))
    has_right = all(keypoint is not None for keypoint in (right_shoulder, right_elbow, right_wrist))
    if not (has_left or has_right):
        return []
    
    hands = []
    #left hand
    if has_left:
        hands.append([
            left_shoulder.x, left_shoulder.y,
            left_elbow.x, left_elbow.y,
            left_wrist.x, left_wrist.y,
            True
        ])
    # right hand
    if has_right:
        hands.append([
            right_shoulder.x, right_shoulder.y,
            right_elbow.x, right_elbow.y,
            right_wrist.x, right_wrist.y,
            False
        ])

    for x1, y1, x2, y2, x3, y3, is_left in hands:
        # pos_hand = pos_wrist + ratio * (pos_wrist - pos_elbox) = (1 + ratio) * pos_wrist - ratio * pos_elbox
        # handRectangle.x = posePtr[wrist*3] + ratioWristElbow * (posePtr[wrist*3] - posePtr[elbow*3]);
        # handRectangle.y = posePtr[wrist*3+1] + ratioWristElbow * (posePtr[wrist*3+1] - posePtr[elbow*3+1]);
        # const auto distanceWristElbow = getDistance(poseKeypoints, person, wrist, elbow);
        # const auto distanceElbowShoulder = getDistance(poseKeypoints, person, elbow, shoulder);
        # handRectangle.width = 1.5f * fastMax(distanceWristElbow, 0.9f * distanceElbowShoulder);
        x = x3 + ratioWristElbow * (x3 - x2)
        y = y3 + ratioWristElbow * (y3 - y2)
        distanceWristElbow = math.sqrt((x3 - x2) ** 2 + (y3 - y2) ** 2)
        distanceElbowShoulder = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        width = 1.5 * max(distanceWristElbow, 0.9 * distanceElbowShoulder)
        # x-y refers to the center --> offset to topLeft point
        # handRectangle.x -= handRectangle.width / 2.f;
        # handRectangle.y -= handRectangle.height / 2.f;
        x -= width / 2
        y -= width / 2  # width = height
        # overflow the image
        if x < 0: x = 0
        if y < 0: y = 0
        width1 = width
        width2 = width
        if x + width > image_width: width1 = image_width - x
        if y + width > image_height: width2 = image_height - y
        width = min(width1, width2)
        # the max hand box value is 20 pixels
        if width >= 20:
            detect_result.append((int(x), int(y), int(width), is_left))

    '''
    return value: [[x, y, w, True if left hand else False]].
    width=height since the network require squared input.
    x, y is the coordinate of top left 
    '''
    return detect_result


# Written by Lvmin
def faceDetect(body: BodyResult, oriImg) -> Union[Tuple[int, int, int], None]:
    """
    Detect the face in the input body pose keypoints and calculate the bounding box for the face.

    Args:
        body (BodyResult): A BodyResult object containing the detected body pose keypoints.
        oriImg (numpy.ndarray): A 3D numpy array representing the original input image.

    Returns:
        Tuple[int, int, int] | None: A tuple containing the coordinates (x, y) of the top-left corner of the
                                   bounding box and the width (height) of the bounding box, or None if the
                                   face is not detected or the bounding box width is less than 20 pixels.

    Notes:
        - The width and height of the bounding box are equal.
        - The minimum bounding box size is 20 pixels.
    """
    # left right eye ear 14 15 16 17
    image_height, image_width = oriImg.shape[0:2]
    
    keypoints = body.keypoints
    head = keypoints[0]
    left_eye = keypoints[14]
    right_eye = keypoints[15]
    left_ear = keypoints[16]
    right_ear = keypoints[17]
    
    if head is None or all(keypoint is None for keypoint in (left_eye, right_eye, left_ear, right_ear)):
        return None

    width = 0.0
    x0, y0 = head.x, head.y

    if left_eye is not None:
        x1, y1 = left_eye.x, left_eye.y
        d = max(abs(x0 - x1), abs(y0 - y1))
        width = max(width, d * 3.0)

    if right_eye is not None:
        x1, y1 = right_eye.x, right_eye.y
        d = max(abs(x0 - x1), abs(y0 - y1))
        width = max(width, d * 3.0)

    if left_ear is not None:
        x1, y1 = left_ear.x, left_ear.y
        d = max(abs(x0 - x1), abs(y0 - y1))
        width = max(width, d * 1.5)

    if right_ear is not None:
        x1, y1 = right_ear.x, right_ear.y
        d = max(abs(x0 - x1), abs(y0 - y1))
        width = max(width, d * 1.5)

    x, y = x0, y0

    x -= width
    y -= width

    if x < 0:
        x = 0

    if y < 0:
        y = 0

    width1 = width * 2
    width2 = width * 2

    if x + width > image_width:
        width1 = image_width - x

    if y + width > image_height:
        width2 = image_height - y

    width = min(width1, width2)

    if width >= 20:
        return int(x), int(y), int(width)
    else:
        return None


# get max index of 2d array
def npmax(array):
    arrayindex = array.argmax(1)
    arrayvalue = array.max(1)
    i = arrayvalue.argmax()
    j = arrayindex[i]
    return i, j

import os
import random

import cv2
import numpy as np
import torch
from pathlib import Path
import warnings
from torch.hub import get_dir, download_url_to_file
from huggingface_hub import hf_hub_download
import tempfile


TORCHHUB_PATH = Path(__file__).parent / 'depth_anything' / 'torchhub'
HF_MODEL_NAME = "lllyasviel/Annotators"
DWPOSE_MODEL_NAME = "yzd-v/DWPose"
BDS_MODEL_NAME = "bdsqlsz/qinglong_controlnet-lllite"
DENSEPOSE_MODEL_NAME = "LayerNorm/DensePose-TorchScript-with-hint-image"
MESH_GRAPHORMER_MODEL_NAME = "hr16/ControlNet-HandRefiner-pruned"
SAM_MODEL_NAME = "dhkim2810/MobileSAM"
UNIMATCH_MODEL_NAME = "hr16/Unimatch"
DEPTH_ANYTHING_MODEL_NAME = "LiheYoung/Depth-Anything" #HF Space
DIFFUSION_EDGE_MODEL_NAME = "hr16/Diffusion-Edge"

temp_dir = tempfile.gettempdir()
annotator_ckpts_path = os.path.join(Path(__file__).parents[2], 'ckpts')
USE_SYMLINKS = False

try:
    annotator_ckpts_path = os.environ['AUX_ANNOTATOR_CKPTS_PATH']
except:
    warnings.warn("Custom pressesor model path not set successfully.")
    pass

try:
    USE_SYMLINKS = eval(os.environ['AUX_USE_SYMLINKS'])
except:
    warnings.warn("USE_SYMLINKS not set successfully. Using default value: False to download models.")
    pass

try:
    temp_dir = os.environ['AUX_TEMP_DIR']
    if len(temp_dir) >= 60:
        warnings.warn(f"custom temp dir is too long. Using default")
        temp_dir = tempfile.gettempdir()
except:
    warnings.warn(f"custom temp dir not set successfully")
    pass

here = Path(__file__).parent.resolve()

def HWC3(x):
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C == 1 or C == 3 or C == 4
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y


def make_noise_disk(H, W, C, F, rng=None):
    if rng:
        noise = rng.uniform(low=0, high=1, size=((H // F) + 2, (W // F) + 2, C))
    else:
        noise = np.random.uniform(low=0, high=1, size=((H // F) + 2, (W // F) + 2, C))
    noise = cv2.resize(noise, (W + 2 * F, H + 2 * F), interpolation=cv2.INTER_CUBIC)
    noise = noise[F: F + H, F: F + W]
    noise -= np.min(noise)
    noise /= np.max(noise)
    if C == 1:
        noise = noise[:, :, None]
    return noise


def nms(x, t, s):
    x = cv2.GaussianBlur(x.astype(np.float32), (0, 0), s)

    f1 = np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]], dtype=np.uint8)
    f2 = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=np.uint8)
    f3 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.uint8)
    f4 = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=np.uint8)

    y = np.zeros_like(x)

    for f in [f1, f2, f3, f4]:
        np.putmask(y, cv2.dilate(x, kernel=f) == x, x)

    z = np.zeros_like(y, dtype=np.uint8)
    z[y > t] = 255
    return z

def min_max_norm(x):
    x -= np.min(x)
    x /= np.maximum(np.max(x), 1e-5)
    return x


def safe_step(x, step=2):
    y = x.astype(np.float32) * float(step + 1)
    y = y.astype(np.int32).astype(np.float32) / float(step)
    return y


def img2mask(img, H, W, low=10, high=90):
    assert img.ndim == 3 or img.ndim == 2
    assert img.dtype == np.uint8

    if img.ndim == 3:
        y = img[:, :, random.randrange(0, img.shape[2])]
    else:
        y = img

    y = cv2.resize(y, (W, H), interpolation=cv2.INTER_CUBIC)

    if random.uniform(0, 1) < 0.5:
        y = 255 - y

    return y < np.percentile(y, random.randrange(low, high))

def safer_memory(x):
    # Fix many MAC/AMD problems
    return np.ascontiguousarray(x.copy()).copy()

UPSCALE_METHODS = ["INTER_NEAREST", "INTER_LINEAR", "INTER_AREA", "INTER_CUBIC", "INTER_LANCZOS4"]
def get_upscale_method(method_str):
    assert method_str in UPSCALE_METHODS, f"Method {method_str} not found in {UPSCALE_METHODS}"
    return getattr(cv2, method_str)

def pad64(x):
    return int(np.ceil(float(x) / 64.0) * 64 - x)

#https://github.com/Mikubill/sd-webui-controlnet/blob/main/scripts/processor.py#L17
#Added upscale_method, mode params
def resize_image_with_pad(input_image, resolution, upscale_method = "", skip_hwc3=False, mode='edge'):
    if skip_hwc3:
        img = input_image
    else:
        img = HWC3(input_image)
    H_raw, W_raw, _ = img.shape
    k = float(resolution) / float(min(H_raw, W_raw))
    H_target = int(np.round(float(H_raw) * k))
    W_target = int(np.round(float(W_raw) * k))
    img = cv2.resize(img, (W_target, H_target), interpolation=get_upscale_method(upscale_method) if k > 1 else cv2.INTER_AREA)
    H_pad, W_pad = pad64(H_target), pad64(W_target)
    img_padded = np.pad(img, [[0, H_pad], [0, W_pad], [0, 0]], mode=mode)

    def remove_pad(x):
        return safer_memory(x[:H_target, :W_target, ...])

    return safer_memory(img_padded), remove_pad
    
def common_input_validate(input_image, output_type, **kwargs):
    if "img" in kwargs:
            warnings.warn("img is deprecated, please use `input_image=...` instead.", DeprecationWarning)
            input_image = kwargs.pop("img")
    
    if "return_pil" in kwargs:
            warnings.warn("return_pil is deprecated. Use output_type instead.", DeprecationWarning)
            output_type = "pil" if kwargs["return_pil"] else "np"
    
    if type(output_type) is bool:
        warnings.warn("Passing `True` or `False` to `output_type` is deprecated and will raise an error in future versions")
        if output_type:
            output_type = "pil"

    if input_image is None:
        raise ValueError("input_image must be defined.")

    if not isinstance(input_image, np.ndarray):
        input_image = np.array(input_image, dtype=np.uint8)
        output_type = output_type or "pil"
    else:
        output_type = output_type or "np"
    
    return (input_image, output_type)

def torch_gc():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def ade_palette():
    """ADE20K palette that maps each class to RGB values."""
    return [[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
            [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
            [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
            [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
            [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
            [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
            [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
            [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
            [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
            [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
            [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
            [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
            [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0],
            [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
            [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255],
            [11, 200, 200], [255, 82, 0], [0, 255, 245], [0, 61, 255],
            [0, 255, 112], [0, 255, 133], [255, 0, 0], [255, 163, 0],
            [255, 102, 0], [194, 255, 0], [0, 143, 255], [51, 255, 0],
            [0, 82, 255], [0, 255, 41], [0, 255, 173], [10, 0, 255],
            [173, 255, 0], [0, 255, 153], [255, 92, 0], [255, 0, 255],
            [255, 0, 245], [255, 0, 102], [255, 173, 0], [255, 0, 20],
            [255, 184, 184], [0, 31, 255], [0, 255, 61], [0, 71, 255],
            [255, 0, 204], [0, 255, 194], [0, 255, 82], [0, 10, 255],
            [0, 112, 255], [51, 0, 255], [0, 194, 255], [0, 122, 255],
            [0, 255, 163], [255, 153, 0], [0, 255, 10], [255, 112, 0],
            [143, 255, 0], [82, 0, 255], [163, 255, 0], [255, 235, 0],
            [8, 184, 170], [133, 0, 255], [0, 255, 92], [184, 0, 255],
            [255, 0, 31], [0, 184, 255], [0, 214, 255], [255, 0, 112],
            [92, 255, 0], [0, 224, 255], [112, 224, 255], [70, 184, 160],
            [163, 0, 255], [153, 0, 255], [71, 255, 0], [255, 0, 163],
            [255, 204, 0], [255, 0, 143], [0, 255, 235], [133, 255, 0],
            [255, 0, 235], [245, 0, 255], [255, 0, 122], [255, 245, 0],
            [10, 190, 212], [214, 255, 0], [0, 204, 255], [20, 0, 255],
            [255, 255, 0], [0, 153, 255], [0, 41, 255], [0, 255, 204],
            [41, 0, 255], [41, 255, 0], [173, 0, 255], [0, 245, 255],
            [71, 0, 255], [122, 0, 255], [0, 255, 184], [0, 92, 255],
            [184, 255, 0], [0, 133, 255], [255, 214, 0], [25, 194, 194],
            [102, 255, 0], [92, 0, 255]]

#https://stackoverflow.com/a/44873382
#Assume that the minimum version of Python ppl use is 3.9
def sha256sum(file_path):
    import hashlib
    h  = hashlib.sha256()
    b  = bytearray(128*1024)
    mv = memoryview(b)
    with open(file_path, 'rb', buffering=0) as f:
        while n := f.readinto(mv):
            h.update(mv[:n])
    return h.hexdigest()

def check_hash_from_torch_hub(file_path, filename):
    basename, _ = filename.split('.')
    _, ref_hash = basename.split('-')
    curr_hash = sha256sum(file_path)
    return curr_hash[:len(ref_hash)] == ref_hash

def custom_torch_download(filename, ckpts_dir=annotator_ckpts_path):
    local_dir = os.path.join(get_dir(), 'checkpoints')
    model_path = os.path.join(local_dir, filename)

    if not os.path.exists(model_path):
        print(f"Failed to find {model_path}.\n Downloading from pytorch.org")
        local_dir = os.path.join(ckpts_dir, "torch")
        if not os.path.exists(local_dir):
            os.mkdir(local_dir)

        model_path = os.path.join(local_dir, filename)

        if not os.path.exists(model_path):
            model_url = "https://download.pytorch.org/models/"+filename
            try:
                download_url_to_file(url = model_url, dst = model_path)
            except:
                warnings.warn(f"SSL verify failed, try use HTTP instead. {filename}'s hash will be checked")
                download_url_to_file(url = model_url, dst = model_path)
                assert check_hash_from_torch_hub(model_path, filename), f"Hash check failed as file {filename} is corrupted"
                print("Hash check passed")
    
    print(f"model_path is {model_path}")
    return model_path

def custom_hf_download(pretrained_model_or_path, filename, cache_dir=temp_dir, ckpts_dir=annotator_ckpts_path, subfolder='', use_symlinks=USE_SYMLINKS, repo_type="model"):
    
    local_dir = os.path.join(ckpts_dir, pretrained_model_or_path)
    model_path = os.path.join(local_dir, *subfolder.split('/'), filename)

    if len(str(model_path)) >= 255:
        warnings.warn(f"Path {model_path} is too long, \n please change annotator_ckpts_path in config.yaml")
    
    if not os.path.exists(model_path):
        print(f"Failed to find {model_path}.\n Downloading from huggingface.co")
        print(f"cacher folder is {cache_dir}, you can change it by custom_tmp_path in config.yaml")
        if use_symlinks:
            cache_dir_d = os.getenv("HUGGINGFACE_HUB_CACHE")
            if cache_dir_d is None:
                import platform
                if platform.system() == "Windows":
                    cache_dir_d = os.path.join(os.getenv("USERPROFILE"), ".cache", "huggingface", "hub")
                else:
                    cache_dir_d = os.path.join(os.getenv("HOME"), ".cache", "huggingface", "hub")
            try:
                # test_link
                if not os.path.exists(cache_dir_d):
                    os.makedirs(cache_dir_d)
                open(os.path.join(cache_dir_d, f"linktest_{filename}.txt"), "w")
                os.link(os.path.join(cache_dir_d, f"linktest_{filename}.txt"), os.path.join(ckpts_dir, f"linktest_{filename}.txt"))
                os.remove(os.path.join(ckpts_dir, f"linktest_{filename}.txt"))
                os.remove(os.path.join(cache_dir_d, f"linktest_{filename}.txt"))
                print("Using symlinks to download models. \n",\
                      "Make sure you have enough space on your cache folder. \n",\
                      "And do not purge the cache folder after downloading.\n",\
                      "Otherwise, you will have to re-download the models every time you run the script.\n",\
                      "You can use USE_SYMLINKS: False in config.yaml to avoid this behavior.")
            except:
                print("Maybe not able to create symlink. Disable using symlinks.")
                use_symlinks = False
                cache_dir_d = os.path.join(cache_dir, "ckpts", pretrained_model_or_path)
        else:
            cache_dir_d = os.path.join(cache_dir, "ckpts", pretrained_model_or_path)

        model_path = hf_hub_download(repo_id=pretrained_model_or_path,
            cache_dir=cache_dir_d,
            local_dir=local_dir,
            subfolder=subfolder,
            filename=filename,
            local_dir_use_symlinks=use_symlinks,
            resume_download=True,
            etag_timeout=100,
            repo_type=repo_type
        )
        if not use_symlinks:
            try:
                import shutil
                shutil.rmtree(os.path.join(cache_dir, "ckpts"))
            except Exception as e :
                print(e)

    print(f"model_path is {model_path}")

    return model_path

