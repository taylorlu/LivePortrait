import torch
import cv2
import numpy as np
import os
from src.config.inference_config import InferenceConfig
from src.config.crop_config import CropConfig
from src.utils.cropper import Cropper
from src.utils.camera import get_rotation_matrix
from src.utils.crop import prepare_paste_back, paste_back
from src.utils.io import load_image_rgb, resize_to_limit
from src.live_portrait_wrapper import LivePortraitWrapper


if __name__ == "__main__":
    inf_cfg = InferenceConfig()
    crop_cfg = CropConfig()
    live_portrait_wrapper = LivePortraitWrapper(inference_cfg=inf_cfg)
    cropper = Cropper(crop_cfg=crop_cfg)
    device = 'cuda:0'
    source = '/beta/workspace/LivePortrait/assets/examples/driving/d19.jpg'
    driving = '/beta/workspace/LivePortrait/assets/examples/driving/7.jpg'

    ######################## step 1: for source
    img_rgb = load_image_rgb(source)
    img_rgb = resize_to_limit(img_rgb, inf_cfg.source_max_dim, inf_cfg.source_division)

    crop_info = cropper.crop_source_image(img_rgb, crop_cfg)
    if crop_info is None:
        raise Exception("No face detected in the source image!")
    source_lmk_crop = crop_info['lmk_crop']
    img_crop_256x256 = crop_info['img_crop_256x256']

    I_s = live_portrait_wrapper.prepare_source(img_crop_256x256)
    x_s_info = live_portrait_wrapper.get_kp_info(I_s)
    x_c_s = x_s_info['kp']
    R_s = get_rotation_matrix(x_s_info['pitch'], x_s_info['yaw'], x_s_info['roll'])
    f_s = live_portrait_wrapper.extract_feature_3d(I_s)
    x_s = live_portrait_wrapper.transform_keypoint(x_s_info)

    mask_ori_float = prepare_paste_back(inf_cfg.mask_crop, crop_info['M_c2o'], dsize=(img_rgb.shape[1], img_rgb.shape[0]))

    ######################## step 2: for driving
    driving_img_rgb = load_image_rgb(driving)
    driving_rgb_crop_256x256 = cv2.resize(driving_img_rgb, (256, 256))
    I_d = live_portrait_wrapper.prepare_source(driving_rgb_crop_256x256)
    x_i_info = live_portrait_wrapper.get_kp_info(I_d)
    exp_d_i = x_i_info['exp']

    ########################
    c_s_eyes_lst, c_s_lip_lst = live_portrait_wrapper.calc_ratio([source_lmk_crop])
    c_s_eyes, c_s_lip = c_s_eyes_lst[0], c_s_lip_lst[0]
    combined_eye_ratio_tensor = live_portrait_wrapper.calc_combined_eye_ratio(c_s_eyes, source_lmk_crop)
    eyes_delta = live_portrait_wrapper.retarget_eye(x_s, combined_eye_ratio_tensor)

    scale_new = x_s_info['scale']

    zero_lip = torch.from_numpy(inf_cfg.lip_array).to(dtype=torch.float32, device=device)

    # delta_new = x_s_info['exp'] + (exp_d_i - zero_lip)
    delta_new = exp_d_i - zero_lip
    # delta_new -= eyes_delta

    # R_s = torch.eye(3)[None, ...].to(device)
    x_d_i_new = scale_new * (x_c_s @ R_s + delta_new)
    # x_d_i_new += eyes_delta

    x_d_i_new = live_portrait_wrapper.stitching(x_s, x_d_i_new)
    out = live_portrait_wrapper.warp_decode(f_s, x_s, x_d_i_new)
    I_p_i = live_portrait_wrapper.parse_output(out['out'])[0]

    I_p_pstbk = paste_back(I_p_i, crop_info['M_c2o'], img_rgb, mask_ori_float)

    cv2.imwrite('__.jpg', I_p_pstbk[..., ::-1])
