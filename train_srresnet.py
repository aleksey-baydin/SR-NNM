import os
import shutil
import time
import torch
from torch import nn
from torch import optim
from torch.cuda import amp
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import config
from dataset import CUDAPrefetcher, TrainValidImageDataset, TestImageDataset
from metrics import PSNR, SSIM
from model import Generator
from progress import AverageMeter, ProgressMeter


def main():
    # Инициализация стартовой эпохи
    start_epoch = 0
    # Инициализация метрик
    best_psnr = 0.0
    best_ssim = 0.0
    train_prefetcher, valid_prefetcher, test_prefetcher = load_dataset()
    print("Load all datasets successfully.")
    model = build_model()
    print("Build SRResNet model successfully.")
    pixel_criterion = define_loss()
    print("Define all loss functions successfully.")
    optimizer = define_optimizer(model)
    print("Define all optimizer functions successfully.")
    scheduler = define_scheduler(optimizer)
    print("Define all optimizer scheduler functions successfully.")
    print("Check whether the pretrained model is restored...")
    if config.resume:
        # Загрузка чекпоинта
        checkpoint = torch.load(config.resume, map_location=lambda storage, loc: storage)
        # Восстановление параметров
        start_epoch = checkpoint["epoch"]
        best_psnr = checkpoint["best_psnr"]
        best_ssim = checkpoint["best_ssim"]
        # Загрузка состояния модели и весов
        model_state_dict = model.state_dict()
        new_state_dict = {k: v for k, v in checkpoint["state_dict"].items() if k in model_state_dict.keys()}
        # Запись весов предобученной модели в текущую
        model_state_dict.update(new_state_dict)
        model.load_state_dict(model_state_dict)
        # Загрузка оптимизатора
        optimizer.load_state_dict(checkpoint["optimizer"])
        # Загрузка планировщика
        scheduler.load_state_dict(checkpoint["scheduler"])
        print("Loaded pretrained model weights.")
    # Создание директории для результатов
    samples_dir = os.path.join("samples", config.exp_name)
    results_dir = os.path.join("results", config.exp_name)
    if not os.path.exists(samples_dir):
        os.makedirs(samples_dir)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    # Создание файла логов
    writer = SummaryWriter(os.path.join("samples", "logs", config.exp_name))
    # Градиентный скейлер
    scaler = amp.GradScaler()
    # Создание моделей оценки
    psnr_model = PSNR(config.upscale_factor, config.only_test_y_channel)
    ssim_model = SSIM(config.upscale_factor, config.only_test_y_channel)
    # Перенос моделей на устройство
    psnr_model = psnr_model.to(device=config.device, memory_format=torch.channels_last, non_blocking=True)
    ssim_model = ssim_model.to(device=config.device, memory_format=torch.channels_last, non_blocking=True)
    t0 = time.time()
    print("SRResNet training started.")
    for epoch in range(start_epoch, config.epochs):
        train(model, train_prefetcher, pixel_criterion, optimizer, epoch, scaler, writer)
        psnr, ssim = validate(model, valid_prefetcher, epoch, writer, psnr_model, ssim_model, "Valid")
        print("\n")
        # Обновление планировщиков
        scheduler.step()
        # Сохранение модели при лучших значениях метрик
        is_best = psnr > best_psnr and ssim > best_ssim
        best_psnr = max(psnr, best_psnr)
        best_ssim = max(ssim, best_ssim)
        torch.save({"epoch": epoch + 1,
                    "best_psnr": best_psnr,
                    "best_ssim": best_ssim,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict()},
                   os.path.join(samples_dir, f"g_epoch_{epoch + 1}.pth.tar"))
        if is_best:
            shutil.copyfile(os.path.join(samples_dir, f"g_epoch_{epoch + 1}.pth.tar"),
                            os.path.join(results_dir, "g_best.pth.tar"))
        if (epoch + 1) == config.epochs:
            shutil.copyfile(os.path.join(samples_dir, f"g_epoch_{epoch + 1}.pth.tar"),
                            os.path.join(results_dir, "g_last.pth.tar"))
    print("SRResNet training has finished.")
    print(f"Epochs: {config.epochs}, time: {(time.time() - t0) / 60:.2f} minutes, best PSNR: {best_psnr}, best SSIM: "
          f"{best_ssim}")


def load_dataset() -> [CUDAPrefetcher, CUDAPrefetcher, CUDAPrefetcher]:
    # Загрузка обучающей, валидационной и тестовой выборки
    train_datasets = TrainValidImageDataset(config.train_image_dir, config.image_size, config.upscale_factor, "Train")
    valid_datasets = TrainValidImageDataset(config.valid_image_dir, config.image_size, config.upscale_factor, "Valid")
    test_datasets = TestImageDataset(config.test_lr_image_dir, config.test_hr_image_dir)
    # Создание даталоадеров
    train_dataloader = DataLoader(train_datasets, batch_size=config.batch_size, shuffle=True,
                                  num_workers=config.num_workers, pin_memory=True, drop_last=True,
                                  persistent_workers=True)
    valid_dataloader = DataLoader(valid_datasets, batch_size=1, shuffle=False, num_workers=1, pin_memory=True,
                                  drop_last=False, persistent_workers=True)
    test_dataloader = DataLoader(test_datasets, batch_size=1, shuffle=False, num_workers=1, pin_memory=True,
                                 drop_last=False, persistent_workers=True)
    # Загрузка данных в лоадеры предобработки
    train_prefetcher = CUDAPrefetcher(train_dataloader, config.device)
    valid_prefetcher = CUDAPrefetcher(valid_dataloader, config.device)
    test_prefetcher = CUDAPrefetcher(test_dataloader, config.device)
    return train_prefetcher, valid_prefetcher, test_prefetcher


def build_model() -> nn.Module:
    model = Generator()
    model = model.to(device=config.device, memory_format=torch.channels_last)
    return model


def define_loss() -> nn.MSELoss:
    pixel_criterion = nn.MSELoss()
    pixel_criterion = pixel_criterion.to(device=config.device, memory_format=torch.channels_last)
    return pixel_criterion


def define_optimizer(model) -> optim.Adam:
    optimizer = optim.Adam(model.parameters(), config.model_lr, config.model_betas)
    return optimizer


def define_scheduler(optimizer: optim.Adam) -> [lr_scheduler.StepLR]:
    scheduler = lr_scheduler.StepLR(optimizer, config.lr_scheduler_step_size, config.lr_scheduler_gamma)
    return scheduler


def train(model, train_prefetcher, pixel_criterion, optimizer, epoch, scaler, writer) -> None:
    # Сколько батчей в одной эпохе
    batches = len(train_prefetcher)
    # Информация об обучении в прогресс-баре
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":6.6f")
    progress = ProgressMeter(batches, [batch_time, data_time, losses], prefix=f"Epoch: [{epoch + 1}]")
    # Переключить состояние модели на "Обучение"
    model.train()
    batch_index = 0
    end = time.time()
    # Инициализация даталоадера и выгрузка первого батча
    train_prefetcher.reset()
    batch_data = train_prefetcher.next()
    while batch_data is not None:
        # Вычисление времени загрузки батча
        data_time.update(time.time() - end)
        # Перенос изображений на выбранное устройство
        lr = batch_data["lr"].to(device=config.device, memory_format=torch.channels_last, non_blocking=True)
        hr = batch_data["hr"].to(device=config.device, memory_format=torch.channels_last, non_blocking=True)
        # Инициализация градиентов генератора
        model.zero_grad(set_to_none=True)
        # Обучение и вычисление функции потерь
        with amp.autocast():
            sr = model(lr)
            loss = pixel_criterion(sr, hr)
        # Обратное распространение ошибки
        scaler.scale(loss).backward()
        # Обновление весов генератора
        scaler.step(optimizer)
        scaler.update()
        # Данные для вывода в терминале
        losses.update(loss.item(), lr.size(0))
        # Вычисление времени, которое требуется для обучения батча
        batch_time.update(time.time() - end)
        end = time.time()
        # Запись данных в логи и терминале
        if batch_index % config.print_frequency == 0:
            writer.add_scalar("Train/Loss", loss.item(), batch_index + epoch * batches + 1)
            progress.display(batch_index)
        # Предзагрузка следующего батча
        batch_data = train_prefetcher.next()
        batch_index += 1


def validate(model, data_prefetcher, epoch, writer, psnr_model, ssim_model, mode) -> [float, float]:
    # Вычисление количества батчей в одной эпохе
    batches = len(data_prefetcher)
    batch_time = AverageMeter("Time", ":6.3f")
    psnres = AverageMeter("PSNR", ":4.2f")
    ssimes = AverageMeter("SSIM", ":4.4f")
    progress = ProgressMeter(batches, [batch_time, psnres, ssimes], prefix=f"{mode}: ")
    # Установка модели в состояние "оценки"
    model.eval()
    # Инициализация индекса текущего батча
    batch_index = 0
    # Инициализация даталоадера и выгрузка первого батча
    data_prefetcher.reset()
    batch_data = data_prefetcher.next()
    # Инициализация времени, требуемого для обработки батча
    end = time.time()
    with torch.no_grad():
        while batch_data is not None:
            # Перенос изображений на выбранное устройство
            lr = batch_data["lr"].to(device=config.device, memory_format=torch.channels_last, non_blocking=True)
            hr = batch_data["hr"].to(device=config.device, memory_format=torch.channels_last, non_blocking=True)
            # Используем генератор для создания фейкового изображения
            with amp.autocast():
                sr = model(lr)
            # Вычисление метрик
            psnr = psnr_model(sr, hr)
            ssim = ssim_model(sr, hr)
            psnres.update(psnr.item(), lr.size(0))
            ssimes.update(ssim.item(), lr.size(0))
            # Вычисление времени, требуемого для обработки батча
            batch_time.update(time.time() - end)
            end = time.time()
            # Отображение результатов в терминале
            if batch_index % (batches // 5) == 0:
                progress.display(batch_index)
            # Предзагрузка следующего батча
            batch_data = data_prefetcher.next()
            batch_index += 1
    # Отображение финальных метрик
    progress.display_summary()
    if mode == "Valid" or mode == "Test":
        writer.add_scalar(f"{mode}/PSNR", psnres.avg, epoch + 1)
        writer.add_scalar(f"{mode}/SSIM", ssimes.avg, epoch + 1)
    else:
        raise ValueError("Unsupported mode, please use `Valid` or `Test`.")

    return psnres.avg, ssimes.avg


if __name__ == "__main__":
    main()
