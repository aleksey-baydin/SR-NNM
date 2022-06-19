import cv2
import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import functional as f
import imgproc


# Функция проверки размеров тензоров
def _check_tensor_shape(raw_tensor: torch.Tensor, dst_tensor: torch.Tensor):
    assert raw_tensor.shape == dst_tensor.shape, \
        f"Supplied images have different sizes {str(raw_tensor.shape)} and {str(dst_tensor.shape)}"


# Имплементация PSNR в PyTorch
def _psnr_torch(raw_tensor: torch.Tensor, dst_tensor: torch.Tensor, crop_border: int,
                only_test_y_channel: bool) -> float:
    # Проверка размеров тензоров
    _check_tensor_shape(raw_tensor, dst_tensor)
    # Кроппинг границ
    if crop_border > 0:
        raw_tensor = raw_tensor[:, :, crop_border:-crop_border, crop_border:-crop_border]
        dst_tensor = dst_tensor[:, :, crop_border:-crop_border, crop_border:-crop_border]
    # Конвертация RGB- в YCbCr-тензор и получение данных только Y-канала
    if only_test_y_channel:
        raw_tensor = imgproc.rgb2ycbcr_torch(raw_tensor, only_use_y_channel=True)
        dst_tensor = imgproc.rgb2ycbcr_torch(dst_tensor, only_use_y_channel=True)
    # Приведение данных к типу torch.float64
    raw_tensor = raw_tensor.to(torch.float64)
    dst_tensor = dst_tensor.to(torch.float64)
    # Вычисление MSE и PSNR по формулам
    mse_value = torch.mean((raw_tensor * 255.0 - dst_tensor * 255.0) ** 2 + 1e-8, dim=[1, 2, 3])
    psnr_metrics = 10 * torch.log10_(255.0 ** 2 / mse_value)
    return psnr_metrics


# Класс метрики PSNR
class PSNR(nn.Module):
    def __init__(self, crop_border: int, only_test_y_channel: bool) -> None:
        super().__init__()
        self.crop_border = crop_border
        self.only_test_y_channel = only_test_y_channel

    def forward(self, raw_tensor: torch.Tensor, dst_tensor: torch.Tensor) -> torch.Tensor:
        psnr_metrics = _psnr_torch(raw_tensor, dst_tensor, self.crop_border, self.only_test_y_channel)
        return psnr_metrics


# Реализация метрики SSIM на PyTorch, где тензоры принимают значение пикселей в диапазоне [0, 255]
def _ssim_torch(raw_tensor: torch.Tensor,
                dst_tensor: torch.Tensor,
                window_size: int,
                gaussian_kernel_window: np.ndarray) -> Tensor:
    # Константы по умолчанию
    c1 = (0.01 * 255.0) ** 2
    c2 = (0.03 * 255.0) ** 2
    # Создание Гауссова окна
    gaussian_kernel_window = torch.from_numpy(gaussian_kernel_window).view(1, 1, window_size, window_size)
    gaussian_kernel_window = gaussian_kernel_window.expand(raw_tensor.size(1), 1, window_size, window_size)
    gaussian_kernel_window = gaussian_kernel_window.to(device=raw_tensor.device, dtype=raw_tensor.dtype)
    # Вычисление средних, дисперсий и ковариации характеристик изображений
    raw_mean = f.conv2d(raw_tensor, gaussian_kernel_window, stride=(1, 1), padding=(0, 0), groups=raw_tensor.shape[1])
    dst_mean = f.conv2d(dst_tensor, gaussian_kernel_window, stride=(1, 1), padding=(0, 0), groups=dst_tensor.shape[1])
    raw_mean_square = raw_mean ** 2
    dst_mean_square = dst_mean ** 2
    raw_dst_mean = raw_mean * dst_mean
    raw_variance = f.conv2d(raw_tensor * raw_tensor, gaussian_kernel_window, stride=(1, 1), padding=(0, 0),
                            groups=raw_tensor.shape[1]) - raw_mean_square
    dst_variance = f.conv2d(dst_tensor * dst_tensor, gaussian_kernel_window, stride=(1, 1), padding=(0, 0),
                            groups=raw_tensor.shape[1]) - dst_mean_square
    raw_dst_covariance = f.conv2d(raw_tensor * dst_tensor, gaussian_kernel_window, stride=1, padding=(0, 0),
                                  groups=raw_tensor.shape[1]) - raw_dst_mean
    # Вычисление числителя и знаменателя по формуле
    ssim_molecular = (2 * raw_dst_mean + c1) * (2 * raw_dst_covariance + c2)
    ssim_denominator = (raw_mean_square + dst_mean_square + c1) * (raw_variance + dst_variance + c2)
    ssim_metrics = ssim_molecular / ssim_denominator
    ssim_metrics = torch.mean(ssim_metrics, [1, 2, 3])
    return ssim_metrics


# Реализация метрики SSIM на PyTorch, где тензоры принимают значение пикселей в диапазоне [0, 255]
def _ssim_single_torch(raw_tensor: torch.Tensor,
                       dst_tensor: torch.Tensor,
                       crop_border: int,
                       only_test_y_channel: bool,
                       window_size: int,
                       gaussian_kernel_window: torch.Tensor) -> torch.Tensor:
    # Проверка размеров тензоров
    _check_tensor_shape(raw_tensor, dst_tensor)
    # Кроппинг границ
    if crop_border > 0:
        raw_tensor = raw_tensor[:, :, crop_border:-crop_border, crop_border:-crop_border]
        dst_tensor = dst_tensor[:, :, crop_border:-crop_border, crop_border:-crop_border]
    # Конвертация RGB- в YCbCr-тензор и получение данных только Y-канала
    if only_test_y_channel:
        raw_tensor = imgproc.rgb2ycbcr_torch(raw_tensor, only_use_y_channel=True)
        dst_tensor = imgproc.rgb2ycbcr_torch(dst_tensor, only_use_y_channel=True)
    # Приведение данных к типу torch.float64
    raw_tensor = raw_tensor.to(torch.float64)
    dst_tensor = dst_tensor.to(torch.float64)
    # Вызов внутренней функции для вычисления метрики
    ssim_metrics = _ssim_torch(raw_tensor * 255.0, dst_tensor * 255.0, window_size, gaussian_kernel_window)
    return ssim_metrics


# Класс метрики SSIM
class SSIM(nn.Module):
    def __init__(self, crop_border: int,
                 only_only_test_y_channel: bool,
                 window_size: int = 11,
                 gaussian_sigma: float = 1.5) -> None:
        super().__init__()
        self.crop_border = crop_border
        self.only_test_y_channel = only_only_test_y_channel
        self.window_size = window_size
        gaussian_kernel = cv2.getGaussianKernel(window_size, gaussian_sigma)
        self.gaussian_kernel_window = np.outer(gaussian_kernel, gaussian_kernel.transpose())

    def forward(self, raw_tensor: torch.Tensor, dst_tensor: torch.Tensor) -> torch.Tensor:
        ssim_metrics = _ssim_single_torch(raw_tensor, dst_tensor, self.crop_border, self.only_test_y_channel,
                                          self.window_size, self.gaussian_kernel_window)
        return ssim_metrics
