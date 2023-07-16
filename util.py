import cv2
import torch
import numpy as np

def img_np_to_tensor(img_np_list):
    out_list = []
    for img_np in img_np_list:
        out_list.append(torch.from_numpy(img_np.astype(np.float32) / 255.0))
    return torch.stack(out_list)
def img_tensor_to_np(img_tensor):
    img_tensor = img_tensor.clone()
    img_tensor = (img_tensor * 255.0).round()
    mask_list = [x.squeeze().numpy().astype(np.uint8) for x in torch.split(img_tensor, 1)]
    return mask_list
    #Thanks ChatGPT

def common_annotator_call(annotator_callback, tensor_image, *args):
    tensor_image_list = img_tensor_to_np(tensor_image)
    out_list = []
    for tensor_image in tensor_image_list:
        call_result = annotator_callback(resize_image(HWC3(tensor_image)), *args)
        H, W, C = tensor_image.shape
        out_list.append(cv2.resize(HWC3(call_result), (W, H), interpolation=cv2.INTER_AREA))
    return out_list

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


def resize_image(input_image, resolution=None):
    H, W, C = input_image.shape
    H = float(H)
    W = float(W)
    k = 0
    if resolution is not None:
        k = float(resolution) / min(H, W)
        H *= k
        W *= k
    H = int(np.round(H / 64.0)) * 64
    W = int(np.round(W / 64.0)) * 64
    img = cv2.resize(input_image, (W, H), interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA)
    return img
