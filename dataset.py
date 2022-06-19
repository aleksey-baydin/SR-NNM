import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import imgproc


class TrainValidImageDataset(Dataset):
    def __init__(self, image_dir: str, image_size: int, upscale_factor: int, mode: str) -> None:
        super(TrainValidImageDataset, self).__init__()
        # Получить названия всех файлов в директории
        self.image_file_names = [os.path.join(image_dir, image_file_name) for image_file_name in os.listdir(image_dir)]
        self.image_size = image_size
        self.upscale_factor = upscale_factor
        # Загрузка обучающей (Train) или валидационной (Valid) выборки
        self.mode = mode

    def __getitem__(self, batch_index: int) -> [torch.Tensor, torch.Tensor]:
        # Чтение батча данных
        image = cv2.imread(self.image_file_names[batch_index], cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
        # Операции обработки
        if self.mode == "Train":
            hr_image = imgproc.random_crop(image, self.image_size)
        elif self.mode == "Valid":
            hr_image = imgproc.center_crop(image, self.image_size)
        else:
            raise ValueError("Unsupported data processing model, please use `Train` or `Valid`.")
        lr_image = imgproc.image_resize(hr_image, 1 / self.upscale_factor)
        # Конвертация в RGB-формат
        lr_image = cv2.cvtColor(lr_image, cv2.COLOR_BGR2RGB)
        hr_image = cv2.cvtColor(hr_image, cv2.COLOR_BGR2RGB)
        # Конвертация в тензоры
        lr_tensor = imgproc.image2tensor(lr_image, range_norm=False, half=False)
        hr_tensor = imgproc.image2tensor(hr_image, range_norm=False, half=False)
        return {"lr": lr_tensor, "hr": hr_tensor}

    def __len__(self) -> int:
        return len(self.image_file_names)


class TestImageDataset(Dataset):
    def __init__(self, test_lr_image_dir: str, test_hr_image_dir: str) -> None:
        super(TestImageDataset, self).__init__()
        # Получить названия всех файлов в директории
        self.lr_image_file_names = [os.path.join(test_lr_image_dir, x) for x in os.listdir(test_lr_image_dir)]
        self.hr_image_file_names = [os.path.join(test_hr_image_dir, x) for x in os.listdir(test_lr_image_dir)]

    def __getitem__(self, batch_index: int) -> [torch.Tensor, torch.Tensor]:
        # Чтение батча данных
        lr_image = cv2.imread(self.lr_image_file_names[batch_index], cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
        hr_image = cv2.imread(self.hr_image_file_names[batch_index], cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
        # Конвертация в RGB-формат
        lr_image = cv2.cvtColor(lr_image, cv2.COLOR_BGR2RGB)
        hr_image = cv2.cvtColor(hr_image, cv2.COLOR_BGR2RGB)
        # Конвертация в тензоры
        lr_tensor = imgproc.image2tensor(lr_image, range_norm=False, half=False)
        hr_tensor = imgproc.image2tensor(hr_image, range_norm=False, half=False)
        return {"lr": lr_tensor, "hr": hr_tensor}

    def __len__(self) -> int:
        return len(self.lr_image_file_names)


class CUDAPrefetcher:
    def __init__(self, dataloader, device: torch.device):
        self.batch_data = None
        self.original_dataloader = dataloader
        self.device = device
        self.data = iter(dataloader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.batch_data = next(self.data)
        except StopIteration:
            self.batch_data = None
            return None
        with torch.cuda.stream(self.stream):
            for k, v in self.batch_data.items():
                if torch.is_tensor(v):
                    self.batch_data[k] = self.batch_data[k].to(self.device, non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch_data = self.batch_data
        self.preload()
        return batch_data

    def reset(self):
        self.data = iter(self.original_dataloader)
        self.preload()

    def __len__(self) -> int:
        return len(self.original_dataloader)
