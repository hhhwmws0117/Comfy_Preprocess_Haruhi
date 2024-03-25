import torch
import time
import os
# import folder_paths
# from PIL import Image
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import numpy as np
from typing import List, NamedTuple, Union
from glob import glob

import cv2
# global variable
#photomaker_path = hf_hub_download(repo_id="TencentARC/PhotoMaker", filename="photomaker-v1.bin", repo_type="model")
device = "cuda" if torch.cuda.is_available() else "cpu"

from open_pose import OpenposeDetector, draw_poses, PoseResult
from util import resize_image_with_pad, common_input_validate, HWC3


import mediapipe as mp

class FaceMesh:
    def __init__(self, face_mesh):
        self.face_mesh = face_mesh
        self.drawing_spec = mp.solutions.drawing_utils.DrawingSpec(thickness=1, circle_radius=1)

    def draw(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image)
        # background = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
        # if results.multi_face_landmarks:
        #     for face_landmarks in results.multi_face_landmarks:
        #         mp.solutions.drawing_utils.draw_landmarks(
        #             image=background,
        #             landmark_list=face_landmarks,
        #             connections=mp.solutions.face_mesh.FACEMESH_FACE_OVAL,
        #             landmark_drawing_spec=None,
        #             connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_contours_style())
        background = np.ones((image.shape[0], image.shape[1], 3), dtype=np.uint8) * 255
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                points = [(int(landmark.x * image.shape[0]), int(landmark.y * image.shape[1])) for landmark in face_landmarks.landmark]
            hull = cv2.convexHull(np.array(points))
            cv2.fillConvexPoly(background, hull, 0)
        return background

face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5)

mesher = FaceMesh(face_mesh)



class LineartStandardDetector:
    def __call__(self, input_image=None, guassian_sigma=6.0, intensity_threshold=8, detect_resolution=512, output_type=None, upscale_method="INTER_CUBIC", **kwargs):
        input_image, output_type = common_input_validate(input_image, output_type, **kwargs)
        input_image, remove_pad = resize_image_with_pad(input_image, detect_resolution, upscale_method)
        
        x = input_image.astype(np.float32)
        g = cv2.GaussianBlur(x, (0, 0), guassian_sigma)
        intensity = np.min(g - x, axis=2).clip(0, 255)
        intensity /= max(16, np.median(intensity[intensity > intensity_threshold]))
        intensity *= 127
        detected_map = intensity.clip(0, 255).astype(np.uint8)
        
        detected_map = HWC3(remove_pad(detected_map))
        if output_type == "pil":
            detected_map = Image.fromarray(detected_map)
        return detected_map
    
lsd = LineartStandardDetector()
openposedet = OpenposeDetector.from_pretrained()




def openImage(image):
    # image = Image.open(img_path)
    oriImg = np.array(image)
    return oriImg

def ensure_dir(file_path):
    """确保文件路径存在，如果不存在，则创建对应的目录"""
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def lsd_img(npImage):
    r_img = lsd(input_image = npImage)
    r_img = Image.fromarray(r_img)
    # save_image_path = f'{save_path}/{image_name}.jpg'
    # ensure_dir(save_image_path)
    # r_img.save(save_image_path)
    return r_img

def getPoses(npImage):
    poses = openposedet.detect_poses(npImage, include_hand=False, include_face=True)
    poses = poses[0]
    return poses

def getFaceKeypoints(poses):
    lst = []
    for point in poses.face:
        lst.append((point.x, point.y))
    return lst


def masked_LineartStandardDetector(origin_image):
    image_mask = mesher.draw(origin_image)
    image_a = Image.fromarray(image_mask)
    image_b = lsd_img(origin_image)

    image_a = image_a.resize((512, 512))
    # 将 image_a 转换为灰度图，然后转换为 numpy 数组
    mask = np.array(image_a.convert('L')) < 128  # 假设黑色像素的值小于128，用于创建遮罩

    # 将 image_b 转换为数组以便操作
    image_b_array = np.array(image_b)

    # 应用遮罩，不想保留的部分设为0
    # 注意：这里假设 image_b 是 RGB 图片，如果是灰度图需要调整
    result_array = np.zeros_like(image_b_array)
    result_array[mask] = image_b_array[mask]

    # 将结果数组转换回图片
    result_image = Image.fromarray(result_array)
    return result_image

# Initialize a blank 512x512 image
# img = Image.new('RGB', (512, 512), 'black')
# draw = ImageDraw.Draw(img)

# Keypoints data redefined
# keypoints = [(0.16992188, 0.5234375), (0.16992188, 0.62597656), (0.19140625, 0.7089844), (0.21191406, 0.7919922), (0.234375, 0.875), (0.31835938, 0.93652344), (0.3828125, 0.9785156), (0.44628906, 0.99902344), (0.5107422, 0.99902344), (0.57421875, 0.9785156), (0.61621094, 0.93652344), (0.6376953, 0.8544922), (0.68066406, 0.7714844), (0.7011719, 0.7089844), (0.72265625, 0.6269531), (0.72265625, 0.5644531), (0.72265625, 0.5019531), (0.234375, 0.4609375), (0.2763672, 0.41992188), (0.3408203, 0.39941406), (0.4248047, 0.39941406), (0.46777344, 0.41992188), (0.57421875, 0.4189453), (0.61621094, 0.3984375), (0.6591797, 0.3779297), (0.7011719, 0.3984375), (0.72265625, 0.41992188), (0.53125, 0.4814453), (0.53125, 0.5234375), (0.5527344, 0.58496094), (0.5527344, 0.6269531), (0.46777344, 0.66796875), (0.5097656, 0.66796875), (0.53125, 0.68847656), (0.5732422, 0.66796875), (0.59472656, 0.66796875), (0.31835938, 0.5019531), (0.36132812, 0.4609375), (0.4033203, 0.4609375), (0.42578125, 0.5019531), (0.4033203, 0.5029297), (0.36132812, 0.5029297), (0.57421875, 0.48242188), (0.61621094, 0.4609375), (0.6582031, 0.4609375), (0.6796875, 0.4814453), (0.6582031, 0.48242188), (0.61621094, 0.48242188), (0.36132812, 0.75097656), (0.4248047, 0.7294922), (0.4892578, 0.7089844), (0.53125, 0.70996094), (0.5732422, 0.7089844), (0.61621094, 0.70996094), (0.6376953, 0.75097656), (0.5957031, 0.8330078), (0.5732422, 0.8955078), (0.53125, 0.8955078), (0.46777344, 0.8955078), (0.40429688, 0.8544922), (0.36132812, 0.75097656), (0.4892578, 0.75), (0.53125, 0.75), (0.5732422, 0.73046875), (0.61621094, 0.75097656), (0.5732422, 0.8339844), (0.53125, 0.8544922), (0.46777344, 0.8544922), (0.38183594, 0.4814453), (0.6171875, 0.4609375), (-0.0009765625, -0.0009765625)]


def draw_colored_keypoints(keypoints, resolution=512):
    # 假设 'keypoints' 是您已经有的所有关键点的坐标列表
    # keypoints = [(x1, y1), (x2, y2), ..., (xn, yn)]

    # 初始化一个空白512x512的图像
    img = Image.new('RGB', (resolution, resolution), 'black')
    draw = ImageDraw.Draw(img)

    # 将关键点坐标转换为像素坐标
    scaled_keypoints = [(x * resolution, y * resolution) for x, y in keypoints]

    # 修改get_color函数，接受一个颜色系基准和索引
    def get_color(base_color, index):
        np.random.seed(index)  # 设置随机数种子，确保对于同一索引，颜色变化是一致的
        variations = np.random.randint(-30, 30, size=3)  # 在给定的范围内生成颜色变化
        # 确保颜色值在0到255之间
        return tuple(np.clip(base_color + variations, 0, 255))

    # 颜色系基准
    color_bases = {
        'red': np.array([255, 50, 50]),
        'blue': np.array([50, 50, 255]),
        'green': np.array([50, 255, 50]),
        'yellow': np.array([255, 255, 50]),
        'purple': np.array([150, 50, 255]),
        'cyan': np.array([50, 255, 255]),
        'orange': np.array([255, 165, 0]),
        'pink': np.array([255, 105, 180]),
        'brown': np.array([165, 42, 42]),
        'grey': np.array([128, 128, 128]),
        'light_blue': np.array([173, 216, 230]),
    }

    line_width = 5

    # 定义线条组和它们的颜色系
    lines_groups = {
        'red': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 0],
        'blue': [36,37,38,39,40,41,36],
        'green': [42,43,44,45,46,47,42],
        'yellow': [27,28,29,30],
        'purple': [31,32,33,34,35],
        'cyan':  [48,60,49,50,51,52,53,64,54],
        'orange': [60,59,58,57,56,55,64],
        'pink': [49,61,62,63,53],
        'brown': [55, 65, 66, 67, 59],
        'grey': [69],
        'light_blue': [68]
    }
    # # 定义线条组和它们的颜色系
    # lines_groups = {
    #     # 每个颜色系对应的线条索引
    # }

    # 根据颜色系和索引顺序绘制线条
    for color_group, indices in lines_groups.items():
        base_color = color_bases[color_group]
        for i in range(len(indices) - 1):
            color = get_color(base_color, i)
            draw.line([scaled_keypoints[indices[i]], scaled_keypoints[indices[i + 1]]], fill=color, width=line_width)
        # 特殊处理只有一个点的情况
        if len(indices) == 1:
            index = indices[0]
            x, y = scaled_keypoints[index]
            radius = line_width // 2
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline=color)
    
    # image_save_path = f'{image_path}/{image_name}.png'
    # ensure_dir(image_save_path)
    # img.save(image_save_path)
    return img

def blend_images(image_a, image_b, alpha=0.2):
    """
    混合两张图片并保存结果
    :param image_path_a: 图片A的路径
    :param image_path_b: 图片B的路径
    :param save_path: 保存路径
    :param save_name: 保存文件名
    :param alpha: 图片A的权重（0到1之间），图片B的权重将是1-alpha
    :return: 混合后的图片对象
    """
    image_a = image_a.convert("RGBA")
    image_b = image_b.convert("RGBA")
    
    # 确保两张图片大小相同
    image_b = image_b.resize(image_a.size)
    
    # 混合图片
    blended_image = Image.blend(image_a, image_b, alpha)
    
    # image_save_path = os.path.join(save_path, f'{save_name}.png')
    # ensure_dir(image_save_path)
    
    # blended_image.save(image_save_path)
    
    return blended_image


import torch
from PIL import Image, ImageOps, ImageSequence
import numpy as np

def convert_images_to_tensor(image_list):
    images = []
    for image_Image in image_list:
        # output_images = []
        # # output_masks = []
        # for i in ImageSequence.Iterator(image_Image):
        #     i = ImageOps.exif_transpose(i)
        #     if i.mode == 'I':
        #         i = i.point(lambda i: i * (1 / 255))
        #     image = i.convert("RGB")
        #     image = np.array(image).astype(np.float32) / 255.0
        #     image = torch.from_numpy(image)[None,]
        #     # if 'A' in i.getbands():
        #     #     mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
        #     #     mask = 1. - torch.from_numpy(mask)
        #     # else:
        #     #     mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
        #     output_images.append(image)
        #     # output_ma/sks.append(mask.unsqueeze(0))

        # if len(output_images) > 1:
        #     output_image = torch.cat(output_images, dim=0)
        #     # output_mask = torch.cat(output_masks, dim=0)
        # else:
        #     output_image = output_images[0]
        #     # output_mask = output_masks[0]
        # output_images_list.append(output_image)
    # for image_path in dir_files:
    #     if limit_images and image_count >= image_load_cap:
    #         break
    #     i = Image.open(image_path)
        i = ImageOps.exif_transpose(image_Image)
        image = i.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        # if 'A' in i.getbands():
        #     mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
        #     mask = 1. - torch.from_numpy(mask)
        #     if not loaded_alpha:
        #         loaded_alpha = True
        #         zero_mask = torch.zeros((len(image[0]),len(image[0][0])), dtype=torch.float32, device="cpu")
        #         masks = [zero_mask] * image_count
        # else:
        #     mask = zero_mask
        images.append(image)
        # masks.append(mask)
        # image_count += 1
    
    if len(images) == 0:
        raise FileNotFoundError(f"No images could be loaded from directory '{directory}'.")

    return torch.cat(images, dim=0)


class ImagePreprocessingNode_mix_v1:
    def __init__(self, ref_image=None):
        self.ref_image = ref_image
        # self.ref_images_path = ref_images_path
        # self.mode = mode

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ref_image": ("IMAGE",)  # 直接输入图像（可选）
                
                # "mode": ([, "path_Input"], {"default": "direct_Input"})  # 选择模式
            },
            "optional": {
                 "resolution": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 64})
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "preprocess_image_mix_v1"
    CATEGORY = "ControlPreprocessHaruhi"
  
    def preprocess_image_mix_v1(self, ref_image=None, resolution=512):
        # 使用传入的参数更新类属性
        # ref_image = ref_image if ref_image is not None else ref_image
        # ref_images_path = ref_images_path if ref_images_path is not None else ref_images_path
        # mode = mode
        # import torchvision.transforms as transforms
        if ref_image is not None:
            # 直接图像处理
            pil_images = []
            for image in ref_image:
                    np_img = openImage(image)
                    np_img_uint8 = np.round(np_img * 255.0).astype(np.uint8)
                    lsd_img_ = lsd_img(np_img_uint8)
                    poses = getPoses(np_img_uint8)
                    keypoints = getFaceKeypoints(poses)
                    colored_img = draw_colored_keypoints(keypoints, resolution=resolution)
                    # colored_img.show()
                    # lsd_img_.show()
                    blended_image = blend_images(colored_img, lsd_img_)
                    # to_tensor = transforms.ToTensor()

                    if keypoints != []:
                        pil_images.append(blended_image)
                    else:
                        raise ValueError("No Faces received")
                    # print(len(pil_images))
            pil_images_tensor = convert_images_to_tensor(pil_images)
            return (pil_images_tensor,)

        else:
            raise ValueError("No Face received")

class ImagePreprocessingNode_maskedlineart_v2:
    def __init__(self, ref_image=None):
        self.ref_image = ref_image
        # self.ref_images_path = ref_images_path
        # self.mode = mode

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ref_image": ("IMAGE",)  # 直接输入图像（可选）
                
                # "mode": ([, "path_Input"], {"default": "direct_Input"})  # 选择模式
            },
            "optional": {
                 "resolution": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 64})
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "preprocess_image_maskedlineart_v2"
    CATEGORY = "ControlPreprocessHaruhi"
  
    def preprocess_image_maskedlineart_v2(self, ref_image=None, resolution=512):
        # 使用传入的参数更新类属性
        # ref_image = ref_image if ref_image is not None else ref_image
        # ref_images_path = ref_images_path if ref_images_path is not None else ref_images_path
        # mode = mode
        # import torchvision.transforms as transforms
        if ref_image is not None:
            # 直接图像处理
            pil_images = []
            for image in ref_image:
                    np_img = openImage(image)
                    np_img_uint8 = np.round(np_img * 255.0).astype(np.uint8)
                    lsd_img_ = masked_LineartStandardDetector(np_img_uint8)
                    poses = getPoses(np_img_uint8)
                    keypoints = getFaceKeypoints(poses)
                    # colored_img = draw_colored_keypoints(keypoints, resolution=resolution)
                    # colored_img.show()
                    # lsd_img_.show()
                    # blended_image = blend_images(colored_img, lsd_img_)
                    # to_tensor = transforms.ToTensor()

                    if keypoints != []:
                        pil_images.append(lsd_img_)
                    else:
                        raise ValueError("No Faces received")
                    # print(len(pil_images))
            pil_images_tensor = convert_images_to_tensor(pil_images)
            return (pil_images_tensor,)

        else:
            raise ValueError("No Face received")
        
class ImagePreprocessingNode_facepose_v2:
    def __init__(self, ref_image=None):
        self.ref_image = ref_image
        # self.ref_images_path = ref_images_path
        # self.mode = mode

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ref_image": ("IMAGE",)  # 直接输入图像（可选）
                
                # "mode": ([, "path_Input"], {"default": "direct_Input"})  # 选择模式
            },
            "optional": {
                 "resolution": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 64})
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "preprocess_image_facepose_v2"
    CATEGORY = "ControlPreprocessHaruhi"
  
    def preprocess_image_facepose_v2(self, ref_image=None, resolution=512):
        # 使用传入的参数更新类属性
        # ref_image = ref_image if ref_image is not None else ref_image
        # ref_images_path = ref_images_path if ref_images_path is not None else ref_images_path
        # mode = mode
        # import torchvision.transforms as transforms
        if ref_image is not None:
            # 直接图像处理
            pil_images = []
            for image in ref_image:
                    np_img = openImage(image)
                    np_img_uint8 = np.round(np_img * 255.0).astype(np.uint8)
                    # lsd_img_ = lsd_img(np_img_uint8)
                    poses = getPoses(np_img_uint8)
                    keypoints = getFaceKeypoints(poses)
                    colored_img = draw_colored_keypoints(keypoints, resolution=resolution)
                    # colored_img.show()
                    # lsd_img_.show()
                    # blended_image = blend_images(colored_img, lsd_img_)
                    # to_tensor = transforms.ToTensor()

                    if keypoints != []:
                        pil_images.append(colored_img)
                    else:
                        raise ValueError("No Faces received")
                    # print(len(pil_images))
            pil_images_tensor = convert_images_to_tensor(pil_images)
            return (pil_images_tensor,)

        else:
            raise ValueError("No Face received")

class ImagePreprocessingNode_mix_v2:
    def __init__(self, ref_image=None):
        self.ref_image = ref_image
        # self.ref_images_path = ref_images_path
        # self.mode = mode

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ref_image": ("IMAGE",)  # 直接输入图像（可选）
                
                # "mode": ([, "path_Input"], {"default": "direct_Input"})  # 选择模式
            },
            "optional": {
                 "resolution": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 64})
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "preprocess_image_mix_v2"
    CATEGORY = "ControlPreprocessHaruhi"
  
    def preprocess_image_mix_v2(self, ref_image=None, resolution=512):
        # 使用传入的参数更新类属性
        # ref_image = ref_image if ref_image is not None else ref_image
        # ref_images_path = ref_images_path if ref_images_path is not None else ref_images_path
        # mode = mode
        # import torchvision.transforms as transforms
        if ref_image is not None:
            # 直接图像处理
            pil_images = []
            for image in ref_image:
                    np_img = openImage(image)
                    np_img_uint8 = np.round(np_img * 255.0).astype(np.uint8)
                    lsd_img_ = masked_LineartStandardDetector(np_img_uint8)
                    poses = getPoses(np_img_uint8)
                    keypoints = getFaceKeypoints(poses)
                    colored_img = draw_colored_keypoints(keypoints, resolution=resolution)
                    # colored_img.show()
                    # lsd_img_.show()
                    blended_image = blend_images(colored_img, lsd_img_)
                    # to_tensor = transforms.ToTensor()

                    if keypoints != []:
                        pil_images.append(blended_image)
                    else:
                        raise ValueError("No Faces received")
                    # print(len(pil_images))
            pil_images_tensor = convert_images_to_tensor(pil_images)
            return (pil_images_tensor,)

        else:
            raise ValueError("No Face received")
        

NODE_CLASS_MAPPINGS = {
    "Ref_Image_Preprocessing_mix_v1": ImagePreprocessingNode_mix_v1,
    "Ref_Image_Preprocessing_facepose_v2": ImagePreprocessingNode_facepose_v2,
    "Ref_Image_Preprocessing_maskedlineart_v2": ImagePreprocessingNode_maskedlineart_v2,
    "Ref_Image_Preprocessing_mix_v2": ImagePreprocessingNode_mix_v2,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Ref_Image_Preprocessing_mix_v1": "📷Ref Image Preprocessing_mix_v1",
    "Ref_Image_Preprocessing_facepose_v2": "📷Ref Image Preprocessing_facepose_v2",
    "Ref_Image_Preprocessing_maskedlineart_v2": "📷Ref Image Preprocessing_maskedlineart_v2",
    "Ref_Image_Preprocessing_mix_v2": "📷Ref Image Preprocessing_mix_v2",
}

