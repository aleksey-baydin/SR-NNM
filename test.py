import os
import cv2
import shutil
import torch
import time
import config
import imgproc
from metrics import PSNR, SSIM
from model import Generator
from progress import AverageMeter, ProgressMeter


def main() -> None:
    # Инициализация модели
    model = Generator().to(device=config.device, memory_format=torch.channels_last)
    print(f"Build {config.exp_name} model successfully.")
    # Загрузка весов
    checkpoint = torch.load(config.model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint["state_dict"])
    print(f"Load {config.exp_name} model weights `{os.path.abspath(config.model_path)}` successfully.")
    # Создание директории для сохранения результатов
    if os.path.exists(config.sr_dir):
        shutil.rmtree(config.sr_dir)
    os.makedirs(config.sr_dir)
    # Переключение модели в состояние "оценивания"
    model.eval()
    # Инициализация моделей метрик
    psnr = PSNR(config.upscale_factor, config.only_test_y_channel)
    ssim = SSIM(config.upscale_factor, config.only_test_y_channel)
    # Отправка моделей на выбранное устройство
    psnr = psnr.to(device=config.device, memory_format=torch.channels_last, non_blocking=True)
    ssim = ssim.to(device=config.device, memory_format=torch.channels_last, non_blocking=True)
    # Инициализация значений метрик нулями
    psnr_metrics = 0.0
    ssim_metrics = 0.0
    # Получение списка изображений тестовой выборки
    file_names = os.listdir(config.lr_dir)
    # Количество изображений тестовой выборки
    total_files = len(file_names)
    # Информация о процессе тестирования в прогресс-баре
    batch_time = AverageMeter("Time", ":6.3f")
    psnres = AverageMeter("PSNR", ":4.2f")
    ssimes = AverageMeter("SSIM", ":4.4f")
    progress = ProgressMeter(total_files, [batch_time, psnres, ssimes], prefix=f"Test: ")
    for index in range(total_files):
        t0 = time.time()
        lr_image_path = os.path.join(config.lr_dir, file_names[index])
        sr_image_path = os.path.join(config.sr_dir, file_names[index])
        hr_image_path = os.path.join(config.hr_dir, file_names[index])
        print(f"Processing `{os.path.abspath(lr_image_path)}`...")
        # Чтение LR- и HR-изображений
        lr_image = cv2.imread(lr_image_path, cv2.IMREAD_UNCHANGED)
        hr_image = cv2.imread(hr_image_path, cv2.IMREAD_UNCHANGED)
        # Конвертация изображений в RGB-формат
        lr_image = cv2.cvtColor(lr_image, cv2.COLOR_BGR2RGB)
        hr_image = cv2.cvtColor(hr_image, cv2.COLOR_BGR2RGB)
        # Конвертация изображений в тензоры
        lr_tensor = imgproc.image2tensor(lr_image, False, False).unsqueeze_(0)
        hr_tensor = imgproc.image2tensor(hr_image, False, False).unsqueeze_(0)
        # Отправка тензоров на выбранное устройство
        lr_tensor = lr_tensor.to(device=config.device, memory_format=torch.channels_last, non_blocking=True)
        hr_tensor = hr_tensor.to(device=config.device, memory_format=torch.channels_last, non_blocking=True)
        with torch.no_grad():
            sr_tensor = model(lr_tensor)
        # Сохранение изображения
        sr_image = imgproc.tensor2image(sr_tensor, False, False)
        sr_image = cv2.cvtColor(sr_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(sr_image_path, sr_image)
        # Вычисление метрик
        psnr_ = psnr(sr_tensor, hr_tensor).item()
        ssim_ = ssim(sr_tensor, hr_tensor).item()
        psnres.update(psnr_)
        ssimes.update(ssim_)
        psnr_metrics += psnr_
        ssim_metrics += ssim_
        # Вычисление времени, которое требуется для обработки одного изображения
        batch_time.update(time.time() - t0)
        # Отображение результатов в терминале
        if index % 5 == 0:
            progress.display(index)
    # Вычисление средних значений метрик
    avg_ssim = 1 if ssim_metrics / total_files > 1 else ssim_metrics / total_files
    avg_psnr = 100 if psnr_metrics / total_files > 100 else psnr_metrics / total_files
    print(f"PSNR: {avg_psnr:4.2f} dB\n"
          f"SSIM: {avg_ssim:4.4f} u")


if __name__ == "__main__":
    main()
