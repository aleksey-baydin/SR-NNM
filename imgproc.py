import math
import random
from typing import Any
import numpy as np
import torch
from torchvision.transforms import functional as f


def image2tensor(image: np.ndarray, range_norm: bool, half: bool) -> torch.Tensor:
    tensor = f.to_tensor(image)
    # Масштабирование из [0, 1] в [-1, 1]
    if range_norm:
        tensor = tensor.mul(2.0).sub(1.0)
    # Конвертация в torch.half
    if half:
        tensor = tensor.half()
    return tensor


def tensor2image(tensor: torch.Tensor, range_norm: bool, half: bool) -> Any:
    if range_norm:
        tensor = tensor.add(1.0).div(2.0)
    if half:
        tensor = tensor.half()
    image = tensor.squeeze(0).permute(1, 2, 0).mul(255).clamp(0, 255).cpu().numpy().astype("uint8")
    return image


# Code reference `https://github.com/xinntao/BasicSR/blob/master/basicsr/utils/matlab_functions.py`
def _cubic(x: Any) -> Any:
    abs_x = torch.abs(x)
    abs_x2 = abs_x ** 2
    abs_x3 = abs_x ** 3
    return (1.5 * abs_x3 - 2.5 * abs_x2 + 1) * ((abs_x <= 1).type_as(abs_x)) + (
            -0.5 * abs_x3 + 2.5 * abs_x2 - 4 * abs_x + 2) * (
               ((abs_x > 1) * (abs_x <= 2)).type_as(abs_x))


# Code reference `https://github.com/xinntao/BasicSR/blob/master/basicsr/utils/matlab_functions.py`
def _calculate_weights_indices(in_length: int,
                               out_length: int,
                               scale: float,
                               kernel_width: int,
                               antialiasing: bool) -> [np.ndarray, np.ndarray, int, int]:
    if (scale < 1) and antialiasing:
        kernel_width = kernel_width / scale
    x = torch.linspace(1, out_length, out_length)
    u = x / scale + 0.5 * (1 - 1 / scale)
    left = torch.floor(u - kernel_width / 2)
    p = math.ceil(kernel_width) + 2
    indices = left.view(out_length, 1).expand(out_length, p) + torch.linspace(0, p - 1, p).view(1, p).expand(
        out_length, p)
    distance_to_center = u.view(out_length, 1).expand(out_length, p) - indices
    if (scale < 1) and antialiasing:
        weights = scale * _cubic(distance_to_center * scale)
    else:
        weights = _cubic(distance_to_center)
    weights_sum = torch.sum(weights, 1).view(out_length, 1)
    weights = weights / weights_sum.expand(out_length, p)
    weights_zero_tmp = torch.sum((weights == 0), 0)
    if not math.isclose(weights_zero_tmp[0], 0, rel_tol=1e-6):
        indices = indices.narrow(1, 1, p - 2)
        weights = weights.narrow(1, 1, p - 2)
    if not math.isclose(weights_zero_tmp[-1], 0, rel_tol=1e-6):
        indices = indices.narrow(1, 0, p - 2)
        weights = weights.narrow(1, 0, p - 2)
    weights = weights.contiguous()
    indices = indices.contiguous()
    sym_len_s = -indices.min() + 1
    sym_len_e = indices.max() - in_length
    indices = indices + sym_len_s - 1
    return weights, indices, int(sym_len_s), int(sym_len_e)


# Code reference `https://github.com/xinntao/BasicSR/blob/master/basicsr/utils/matlab_functions.py`
def image_resize(image: Any, scale_factor: float, antialiasing: bool = True) -> Any:
    squeeze_flag = False
    if type(image).__module__ == np.__name__:
        numpy_type = True
        if image.ndim == 2:
            image = image[:, :, None]
            squeeze_flag = True
        image = torch.from_numpy(image.transpose(2, 0, 1)).float()
    else:
        numpy_type = False
        if image.ndim == 2:
            image = image.unsqueeze(0)
            squeeze_flag = True
    in_c, in_h, in_w = image.size()
    out_h, out_w = math.ceil(in_h * scale_factor), math.ceil(in_w * scale_factor)
    kernel_width = 4
    weights_h, indices_h, sym_len_hs, sym_len_he = _calculate_weights_indices(in_h, out_h, scale_factor, kernel_width,
                                                                              antialiasing)
    weights_w, indices_w, sym_len_ws, sym_len_we = _calculate_weights_indices(in_w, out_w, scale_factor, kernel_width,
                                                                              antialiasing)
    img_aug = torch.FloatTensor(in_c, in_h + sym_len_hs + sym_len_he, in_w)
    img_aug.narrow(1, sym_len_hs, in_h).copy_(image)
    sym_patch = image[:, :sym_len_hs, :]
    inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(1, inv_idx)
    img_aug.narrow(1, 0, sym_len_hs).copy_(sym_patch_inv)
    sym_patch = image[:, -sym_len_he:, :]
    inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(1, inv_idx)
    img_aug.narrow(1, sym_len_hs + in_h, sym_len_he).copy_(sym_patch_inv)
    out_1 = torch.FloatTensor(in_c, out_h, in_w)
    kernel_width = weights_h.size(1)
    for i in range(out_h):
        idx = int(indices_h[i][0])
        for j in range(in_c):
            out_1[j, i, :] = img_aug[j, idx:idx + kernel_width, :].transpose(0, 1).mv(weights_h[i])
    out_1_aug = torch.FloatTensor(in_c, out_h, in_w + sym_len_ws + sym_len_we)
    out_1_aug.narrow(2, sym_len_ws, in_w).copy_(out_1)
    sym_patch = out_1[:, :, :sym_len_ws]
    inv_idx = torch.arange(sym_patch.size(2) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(2, inv_idx)
    out_1_aug.narrow(2, 0, sym_len_ws).copy_(sym_patch_inv)
    sym_patch = out_1[:, :, -sym_len_we:]
    inv_idx = torch.arange(sym_patch.size(2) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(2, inv_idx)
    out_1_aug.narrow(2, sym_len_ws + in_w, sym_len_we).copy_(sym_patch_inv)
    out_2 = torch.FloatTensor(in_c, out_h, out_w)
    kernel_width = weights_w.size(1)
    for i in range(out_w):
        idx = int(indices_w[i][0])
        for j in range(in_c):
            out_2[j, :, i] = out_1_aug[j, :, idx:idx + kernel_width].mv(weights_w[i])
    if squeeze_flag:
        out_2 = out_2.squeeze(0)
    if numpy_type:
        out_2 = out_2.numpy()
        if not squeeze_flag:
            out_2 = out_2.transpose(1, 2, 0)
    return out_2


def rgb2ycbcr_torch(tensor: torch.Tensor, only_use_y_channel: bool) -> torch.Tensor:
    if only_use_y_channel:
        weight = torch.Tensor([[65.481], [128.553], [24.966]]).to(tensor)
        tensor = torch.matmul(tensor.permute(0, 2, 3, 1), weight).permute(0, 3, 1, 2) + 16.0
    else:
        weight = torch.Tensor([[65.481, -37.797, 112.0],
                               [128.553, -74.203, -93.786],
                               [24.966, 112.0, -18.214]]).to(tensor)
        bias = torch.Tensor([16, 128, 128]).view(1, 3, 1, 1).to(tensor)
        tensor = torch.matmul(tensor.permute(0, 2, 3, 1), weight).permute(0, 3, 1, 2) + bias
    tensor /= 255.
    return tensor


def center_crop(image: np.ndarray, image_size: int) -> np.ndarray:
    image_height, image_width = image.shape[:2]
    top = (image_height - image_size) // 2
    left = (image_width - image_size) // 2
    patch_image = image[top:top + image_size, left:left + image_size, ...]
    return patch_image


def random_crop(image: np.ndarray, image_size: int) -> np.ndarray:
    image_height, image_width = image.shape[:2]
    top = random.randint(0, image_height - image_size)
    left = random.randint(0, image_width - image_size)
    patch_image = image[top:top + image_size, left:left + image_size, ...]
    return patch_image
