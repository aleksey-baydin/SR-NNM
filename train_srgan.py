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
from model import Discriminator, Generator, ContentLoss
from progress import AverageMeter, ProgressMeter


def main():
    # Инициализация стартовой эпохи
    start_epoch = 0
    # Инициализация метрик
    best_psnr = 0.0
    best_ssim = 0.0
    train_prefetcher, valid_prefetcher, test_prefetcher = load_dataset()
    print("Load all datasets successfully.")
    discriminator, generator = build_model()
    print("Build SRGAN model successfully.")
    content_criterion, adversarial_criterion = define_loss()
    print("Define all loss functions successfully.")
    d_optimizer, g_optimizer = define_optimizer(discriminator, generator)
    print("Define all optimizer functions successfully.")
    d_scheduler, g_scheduler = define_scheduler(d_optimizer, g_optimizer)
    print("Define all optimizer scheduler functions successfully.")
    if config.resume:
        print("Loading SRResNet model weights")
        # Загрузка чекпоинта
        checkpoint = torch.load(config.resume, map_location=lambda storage, loc: storage)
        generator.load_state_dict(checkpoint["state_dict"])
        print("Loaded SRResNet model weights.")
    print("Check whether the pretrained discriminator model is restored...")
    if config.resume_d:
        # Загрузка чекпоинта
        checkpoint = torch.load(config.resume_d, map_location=lambda storage, loc: storage)
        # Восстановление параметров
        start_epoch = checkpoint["epoch"]
        best_psnr = checkpoint["best_psnr"]
        best_ssim = checkpoint["best_ssim"]
        # Загрузка состояния дискриминатора и весов
        model_state_dict = discriminator.state_dict()
        new_state_dict = {k: v for k, v in checkpoint["state_dict"].items() if k in model_state_dict.keys()}
        # Запись весов предобученной модели в текущую
        model_state_dict.update(new_state_dict)
        discriminator.load_state_dict(model_state_dict)
        # Загрузка оптимизатора
        d_optimizer.load_state_dict(checkpoint["optimizer"])
        # Загрузка планировщика
        d_scheduler.load_state_dict(checkpoint["scheduler"])
        print("Loaded pretrained discriminator model weights.")
    print("Check whether the pretrained generator model is restored...")
    if config.resume_g:
        # Загрузка чекпоинта
        checkpoint = torch.load(config.resume_g, map_location=lambda storage, loc: storage)
        # Восстановление параметров
        config.start_epoch = checkpoint["epoch"]
        best_psnr = checkpoint["best_psnr"]
        best_ssim = checkpoint["best_ssim"]
        # Загрузка состояния генератора и весов
        model_state_dict = generator.state_dict()
        new_state_dict = {k: v for k, v in checkpoint["state_dict"].items() if k in model_state_dict.keys()}
        # Запись весов предобученной модели в текущую
        model_state_dict.update(new_state_dict)
        generator.load_state_dict(model_state_dict)
        # Загрузка оптимизатора
        g_optimizer.load_state_dict(checkpoint["optimizer"])
        # Загрузка планировщика
        g_scheduler.load_state_dict(checkpoint["scheduler"])
        print("Loaded pretrained generator model weights.")
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
    print("SRGAN training started.")
    for epoch in range(start_epoch, config.epochs):
        train(discriminator,
              generator,
              train_prefetcher,
              content_criterion,
              adversarial_criterion,
              d_optimizer,
              g_optimizer,
              epoch,
              scaler,
              writer)
        psnr, ssim = validate(generator, valid_prefetcher, epoch, writer, psnr_model, ssim_model, "Valid")
        print("\n")
        # Обновление планировщиков
        d_scheduler.step()
        g_scheduler.step()
        # Сохранение модели при лучших значениях метрик
        is_best = psnr > best_psnr and ssim > best_ssim
        best_psnr = max(psnr, best_psnr)
        best_ssim = max(ssim, best_ssim)
        torch.save({"epoch": epoch + 1,
                    "best_psnr": best_psnr,
                    "best_ssim": best_ssim,
                    "state_dict": discriminator.state_dict(),
                    "optimizer": d_optimizer.state_dict(),
                    "scheduler": d_scheduler.state_dict()},
                   os.path.join(samples_dir, f"d_epoch_{epoch + 1}.pth.tar"))
        torch.save({"epoch": epoch + 1,
                    "best_psnr": best_psnr,
                    "best_ssim": best_ssim,
                    "state_dict": generator.state_dict(),
                    "optimizer": g_optimizer.state_dict(),
                    "scheduler": g_scheduler.state_dict()},
                   os.path.join(samples_dir, f"g_epoch_{epoch + 1}.pth.tar"))
        if is_best:
            shutil.copyfile(os.path.join(samples_dir, f"d_epoch_{epoch + 1}.pth.tar"),
                            os.path.join(results_dir, "d_best.pth.tar"))
            shutil.copyfile(os.path.join(samples_dir, f"g_epoch_{epoch + 1}.pth.tar"),
                            os.path.join(results_dir, "g_best.pth.tar"))
        if (epoch + 1) == config.epochs:
            shutil.copyfile(os.path.join(samples_dir, f"d_epoch_{epoch + 1}.pth.tar"),
                            os.path.join(results_dir, "d_last.pth.tar"))
            shutil.copyfile(os.path.join(samples_dir, f"g_epoch_{epoch + 1}.pth.tar"),
                            os.path.join(results_dir, "g_last.pth.tar"))
    print("SRGAN training has finished.")
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


def build_model() -> [nn.Module, nn.Module]:
    discriminator = Discriminator()
    generator = Generator()
    discriminator = discriminator.to(device=config.device, memory_format=torch.channels_last)
    generator = generator.to(device=config.device, memory_format=torch.channels_last)
    return discriminator, generator


def define_loss() -> [ContentLoss, nn.BCEWithLogitsLoss]:
    content_criterion = ContentLoss(config.feature_model_extractor_node,
                                    config.feature_model_normalize_mean,
                                    config.feature_model_normalize_std)
    adversarial_criterion = nn.BCEWithLogitsLoss()
    content_criterion = content_criterion.to(device=config.device, memory_format=torch.channels_last)
    adversarial_criterion = adversarial_criterion.to(device=config.device, memory_format=torch.channels_last)
    return content_criterion, adversarial_criterion


def define_optimizer(discriminator: nn.Module, generator: nn.Module) -> [optim.Adam, optim.Adam]:
    d_optimizer = optim.Adam(discriminator.parameters(), config.model_lr, config.model_betas)
    g_optimizer = optim.Adam(generator.parameters(), config.model_lr, config.model_betas)
    return d_optimizer, g_optimizer


def define_scheduler(d_optimizer: optim.Adam, g_optimizer: optim.Adam) -> [lr_scheduler.StepLR, lr_scheduler.StepLR]:
    d_scheduler = lr_scheduler.StepLR(d_optimizer, config.lr_scheduler_step_size, config.lr_scheduler_gamma)
    g_scheduler = lr_scheduler.StepLR(g_optimizer, config.lr_scheduler_step_size, config.lr_scheduler_gamma)
    return d_scheduler, g_scheduler


def train(discriminator,
          generator,
          train_prefetcher,
          content_criterion,
          adversarial_criterion,
          d_optimizer,
          g_optimizer,
          epoch,
          scaler,
          writer) -> None:
    # Сколько итераций при обработке одного батча
    batches = len(train_prefetcher)
    # Информация об обучении в прогресс-баре
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    content_losses = AverageMeter("Content loss", ":6.6f")
    adversarial_losses = AverageMeter("Adversarial loss", ":6.6f")
    d_hr_probabilities = AverageMeter("D(HR)", ":6.3f")
    d_sr_probabilities = AverageMeter("D(SR)", ":6.3f")
    progress = ProgressMeter(batches,
                             [batch_time, data_time,
                              content_losses, adversarial_losses,
                              d_hr_probabilities, d_sr_probabilities],
                             prefix=f"Epoch: [{epoch + 1}]")
    # Переключить состояние модели на "Обучение"
    discriminator.train()
    generator.train()
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
        # Используется для вывода дискриминатора при бинарной классификации: выборка из исходного датасета помечается 1,
        # а сгенерированная выборка помечается 0
        real_label = torch.full([lr.size(0), 1], 1.0, dtype=lr.dtype, device=config.device)
        fake_label = torch.full([lr.size(0), 1], 0.0, dtype=lr.dtype, device=config.device)
        # Начало обучения дискриминатора
        # Включение обратного распространения ошибки дискриминатора
        for d_parameters in discriminator.parameters():
            d_parameters.requires_grad = True
        # Инициализация градиентов дискриминатора
        discriminator.zero_grad(set_to_none=True)
        # Вычисление значения дискриминатора для реальной выборки
        with amp.autocast():
            hr_output = discriminator(hr)
            d_loss_hr = adversarial_criterion(hr_output, real_label)
        # Вызов функции изменения градиента на значение функции потерь от реальной выборки
        scaler.scale(d_loss_hr).backward()
        # Вычисление значения дискриминатора для сгенерированной выборки
        with amp.autocast():
            # Используем генератор для получения сгенерированных изображений
            sr = generator(lr)
            sr_output = discriminator(sr.detach().clone())
            d_loss_sr = adversarial_criterion(sr_output, fake_label)
            # Вычисление суммы функций потерь
            d_loss = d_loss_sr + d_loss_hr
        # Вызов функции изменения градиента на значение функции потерь от сгенерированной выборки
        scaler.scale(d_loss_sr).backward()
        # Шаг оптимизатора - дискриминатор пытается лучше определить, какое из представленных изображений является
        # реальным, а какое - сгенерированным
        scaler.step(d_optimizer)
        scaler.update()
        # Заканчиваем шаг обучения дискриминатора

        # Начинаем обучать генератор
        # Выключение обратного распространения ошибки дискриминатора
        for d_parameters in discriminator.parameters():
            d_parameters.requires_grad = False
        # Инициализация градиентов генератора
        generator.zero_grad(set_to_none=True)
        # Вычисление функций потерь генератора
        with amp.autocast():
            content_loss = config.content_weight * content_criterion(sr, hr)
            adversarial_loss = config.adversarial_weight * adversarial_criterion(discriminator(sr), real_label)
            # Вычисление суммы функций потерь
            g_loss = content_loss + adversarial_loss
        # Вызов функции изменения градиента на значение функции потерь от сгенерированной выборки
        scaler.scale(g_loss).backward()
        # Шаг оптимизатора - генератор пытается обмануть дискриминатор, создавая более реалистичные изображения
        scaler.step(g_optimizer)
        scaler.update()
        # Заканчиваем шаг обучения генератора

        # Вычисление значения дискриминатора на реальной и сгенерированной выборке
        d_hr_probability = torch.sigmoid_(torch.mean(hr_output.detach()))
        d_sr_probability = torch.sigmoid_(torch.mean(sr_output.detach()))
        # Данные для вывода в терминале
        content_losses.update(content_loss.item(), lr.size(0))
        adversarial_losses.update(adversarial_loss.item(), lr.size(0))
        d_hr_probabilities.update(d_hr_probability.item(), lr.size(0))
        d_sr_probabilities.update(d_sr_probability.item(), lr.size(0))
        # Вычисление времени, которое требуется для обучения батча
        batch_time.update(time.time() - end)
        end = time.time()
        # Запись данных в логи
        if batch_index % config.print_frequency == 0:
            iters = batch_index + epoch * batches + 1
            writer.add_scalar("Train/D_Loss", d_loss.item(), iters)
            writer.add_scalar("Train/G_Loss", g_loss.item(), iters)
            writer.add_scalar("Train/Content_Loss", content_loss.item(), iters)
            writer.add_scalar("Train/Adversarial_Loss", adversarial_loss.item(), iters)
            writer.add_scalar("Train/D(HR)_Probability", d_hr_probability.item(), iters)
            writer.add_scalar("Train/D(SR)_Probability", d_sr_probability.item(), iters)
            progress.display(batch_index)
        # Предзагрузка следующего батча
        batch_data = train_prefetcher.next()
        batch_index += 1


def validate(model,
             data_prefetcher,
             epoch,
             writer,
             psnr_model,
             ssim_model,
             mode) -> [float, float]:
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
