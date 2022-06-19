import random
import numpy as np
import torch
from torch.backends import cudnn


# Установка случайного ядра
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)
# Использовать GPU по умолчанию
device = torch.device("cuda", 0)
# Включение этого параметра, когда размер изображения не меняется при обучении, может ускорить процесс
cudnn.benchmark = True
# При оценке SR моделей использовать только Y-канал изображений
only_test_y_channel = True
# Фактор увеличения разрешения
upscale_factor = 4
# Текущий метод обучения
mode = "train_srresnet"
# Имя модели (для создания директории с результатами и логами)
exp_name = "SRResNet_v1"
# Пути до датасета
train_image_dir = f"data/pleural_cells/{exp_name}/train"
valid_image_dir = f"data/pleural_cells/{exp_name}/valid"
test_lr_image_dir = f"data/pleural_cells/test/LRbicx{upscale_factor}"
test_hr_image_dir = f"data/pleural_cells/test/GTmod12"
# Настройки изображения и даталоадера
image_size = 96
batch_size = 16
num_workers = 4
# Параметры предобученной модели
start_epoch = 0
resume = ""
resume_d = ""
resume_g = ""
# Количество эпох обучения
epochs = 20 if mode == "train_srresnet" else 12
# Параметры оптимизатора
model_lr = 1e-4
model_betas = (0.9, 0.999)
# Частота вывода результатов в терминале
print_frequency = 200
# Конфигурация слоя предобученной модели VGG
feature_model_extractor_node = "features.35"
feature_model_normalize_mean = [0.485, 0.456, 0.406]
feature_model_normalize_std = [0.229, 0.224, 0.225]
# Вес функции потерь
content_weight = 1.0
adversarial_weight = 0.001
# Параметры LR (learning rate) планировщика
lr_scheduler_step_size = epochs // 2
lr_scheduler_gamma = 0.1
# Пути до директорий (тест)
lr_dir = f"data/pleural_cells/test/LRbicx{upscale_factor}"
sr_dir = f"results/sr_images"
hr_dir = f"data/pleural_cells/test/GTmod12"
# Путь до модели
model_path = f"results/{exp_name}/g_best.pth.tar"
# Пути для предсказания
predict_image_dir = f"data/pleural_cells/test/LRbicx{upscale_factor}"
output_image_dir = f"results/predicted_sr_images"
