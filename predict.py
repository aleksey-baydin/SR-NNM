import torch
import os
import cv2
import numpy as np
import config
from model import Generator
import imgproc


def main() -> None:
    model = Generator().to(device=config.device)
    # Загрузка предобученной модели
    checkpoint = torch.load(config.model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint["state_dict"])
    print(f"Load SRGAN model weights `{os.path.abspath(config.model_path)}` successfully.")
    # Создание директории для сохранения результатов
    if not os.path.exists(config.output_image_dir):
        os.makedirs(config.output_image_dir)
    model.eval()
    model.half()
    file_names = os.listdir(config.predict_image_dir)
    print("Prediction started")
    for filename in file_names:
        lr_image_path = os.path.join(config.predict_image_dir, filename)
        sr_image_path = os.path.join(config.output_image_dir, filename)
        print(f"Processing `{lr_image_path}`...")
        # Чтение изображения из файла и конвертация в np.array
        lr_image = cv2.imread(lr_image_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0
        # Конвертация в RGB
        lr_image = cv2.cvtColor(lr_image, cv2.COLOR_BGR2RGB)
        # Представление в виде тензора
        lr_tensor = imgproc.image2tensor(lr_image, range_norm=False, half=True).unsqueeze_(0)
        # Отправка изображения на устройство
        lr_tensor = lr_tensor.to(device=config.device, memory_format=torch.channels_last, non_blocking=True)
        with torch.no_grad():
            sr_tensor = model(lr_tensor)
        # Обратная конвертация из тензора в изображение и далее в BGR-формат
        sr_image = imgproc.tensor2image(sr_tensor, range_norm=False, half=True)
        sr_image = cv2.cvtColor(sr_image, cv2.COLOR_RGB2BGR)
        # Сохранение SR-изображения
        cv2.imwrite(sr_image_path, sr_image)
    print("Super-resolution processing is over")


if __name__ == "__main__":
    main()
