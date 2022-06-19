import torch
from torch import nn
from torch.nn import functional as f
from torchvision import models
from torchvision import transforms
from torchvision.models.feature_extraction import create_feature_extractor

import config


class ResidualConvBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super(ResidualConvBlock, self).__init__()
        self.rcb = nn.Sequential(
            nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
            nn.Conv2d(channels, channels, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.rcb(x)
        out = torch.add(out, identity)
        return out


class UpsampleBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super(UpsampleBlock, self).__init__()
        self.upsample_block = nn.Sequential(
            nn.Conv2d(channels, channels * 4, (3, 3), (1, 1), (1, 1)),
            nn.PixelShuffle(2),
            nn.PReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.upsample_block(x)
        return out


class Discriminator(nn.Module):
    def __init__(self) -> None:
        super(Discriminator, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, (3, 3), (1, 1), (1, 1), bias=True),  # (3) x 96 x 96 (c.image_size x c.image.size)
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 64, (3, 3), (2, 2), (1, 1), bias=False),
            # (64) x 48 x 48 (c.image_size // 2 x c.image.size // 2)
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(64, 128, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 128, (3, 3), (2, 2), (1, 1), bias=False),  # (128) x 24 x 24
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 256, (3, 3), (2, 2), (1, 1), bias=False),  # (256) x 12 x 12
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 512, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(512, 512, (3, 3), (2, 2), (1, 1), bias=False),  # (512) x 6 x 6
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
        )
        out_size = config.image_size // 16
        self.classifier = nn.Sequential(
            nn.Linear(512 * out_size * out_size, 1024),
            nn.LeakyReLU(0.2, True),
            nn.Linear(1024, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.features(x)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out


class Generator(nn.Module):
    def __init__(self) -> None:
        super(Generator, self).__init__()
        # Первый сверточный слой
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 64, (9, 9), (1, 1), (4, 4)),
            nn.PReLU(),
        )
        # Остаточные блоки
        trunk = []
        for _ in range(16):
            trunk.append(ResidualConvBlock(64))
        self.trunk = nn.Sequential(*trunk)
        # Второй сверточный слой
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1), bias=False),
            nn.BatchNorm2d(64),
        )
        # Блок апскейлинга
        upsampling = []
        for _ in range(2):
            upsampling.append(UpsampleBlock(64))
        self.upsampling = nn.Sequential(*upsampling)
        # Выходной слой
        self.conv_block3 = nn.Conv2d(64, 3, (9, 9), (1, 1), (4, 4))
        # Инициализация весов
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        out1 = self.conv_block1(x)
        out = self.trunk(out1)
        out2 = self.conv_block2(out)
        out = torch.add(out1, out2)
        out = self.upsampling(out)
        out = self.conv_block3(out)
        out = torch.clamp_(out, 0.0, 1.0)
        return out

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)


class ContentLoss(nn.Module):
    def __init__(self, feature_model_extractor_node: str,
                 feature_model_normalize_mean: list,
                 feature_model_normalize_std: list) -> None:
        super(ContentLoss, self).__init__()
        self.feature_model_extractor_node = feature_model_extractor_node
        model = models.vgg19(True)
        self.feature_extractor = create_feature_extractor(model, [feature_model_extractor_node])
        self.feature_extractor.eval()
        self.normalize = transforms.Normalize(feature_model_normalize_mean, feature_model_normalize_std)
        # «Заморозить» параметры модели VGG
        for model_parameters in self.feature_extractor.parameters():
            model_parameters.requires_grad = False

    def forward(self, sr_tensor: torch.Tensor, hr_tensor: torch.Tensor) -> torch.Tensor:
        sr_tensor = self.normalize(sr_tensor)
        hr_tensor = self.normalize(hr_tensor)
        sr_feature = self.feature_extractor(sr_tensor)[self.feature_model_extractor_node]
        hr_feature = self.feature_extractor(hr_tensor)[self.feature_model_extractor_node]
        content_loss = f.mse_loss(sr_feature, hr_feature)
        return content_loss
