import cv2
import time
import matplotlib.pyplot as plt
import os
import math
import random
import argparse
import numpy as np


def calc_psnr(orig_img, interp_img):
    # Сплит по каналам изображения
    orig_b, orig_g, orig_r = cv2.split(orig_img)
    interp_b, interp_g, interp_r = cv2.split(interp_img)
    # Вычисление среднеквадратичной ошибки по каждому из каналов отдельно
    mse_b = ((orig_b - interp_b) ** 2).sum()
    mse_g = ((orig_g - interp_g) ** 2).sum()
    mse_r = ((orig_r - interp_r) ** 2).sum()
    h, w, c = orig_img.shape
    # Вычисление общей MSE
    mse = (mse_b + mse_g + mse_r) / (3 * h * w)
    psnr = 10 * math.log10(255 ** 2 / mse)
    return round(psnr, 2)


def calc_ssim(img1, img2):
    # Константы по умолчанию (значения пикселей img1 и img2 в диапазоне [0;1])
    c1 = 0.01**2
    c2 = 0.03**2
    # Ядро и окно для свертки
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    # Средние значения
    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1*mu2
    # Дисперсии и значение ковариации
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
    # Вычисление метрики по формуле
    ssim_index = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return ssim_index.mean()


def interpolate(orig_img, factor, interp_algorithm=cv2.INTER_LINEAR):
    h, w, c = orig_img.shape
    h_, w_ = int(h * factor), int(w * factor)
    return cv2.resize(orig_img, (w_, h_), interpolation=interp_algorithm)


def calculate_sample_avg_psnr(sample_path, hr_sample_path, factor, interp_algorithm=cv2.INTER_LINEAR):
    total_psnr = 0.0
    images = os.listdir(sample_path)
    t0 = time.time()
    for image in images:
        orig_path = os.path.join(sample_path, image)
        hr_path = os.path.join(hr_sample_path, image)
        orig_img = cv2.imread(orig_path)
        interp_img = interpolate(orig_img, factor, interp_algorithm)
        hr_img = cv2.imread(hr_path)
        total_psnr += calc_psnr(hr_img, interp_img)
    return round(total_psnr / len(images), 2), round((time.time() - t0), 2)


def print_validation_results(interp_method, _psnr, _time):
    print(f"{interp_method} Interpolation: PSNR={_psnr}, time {_time} ms")


def save_interpolation_result(img_path, out_path, factor, algorithms, psnres):
    img = cv2.imread(img_path)
    images = [interpolate(img, factor, alg[1]) for alg in algorithms]
    fig = plt.figure(figsize=(20, 16))
    h = int((len(images) + 1) / 2)
    plt.subplot(2, h, 1)
    plt.title('Original', fontdict={'fontsize': 20})
    plt.imshow(img)
    plt.axis('off')
    for i in range(len(images)):
        alg = algorithms[i][0]
        plt.subplot(2, h, i + 2)
        title = f"{alg} PSNR={psnres[alg]}"
        plt.title(title, fontdict={'fontsize': 20})
        plt.imshow(images[i])
        plt.axis('off')
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.savefig(out_path)
    plt.close(fig)


def main(_args):
    interpolation_algorithms = [
        ("Nearest Neighbour", cv2.INTER_NEAREST),
        ("Bilinear", cv2.INTER_LINEAR),
        ("Bicubic", cv2.INTER_CUBIC)
    ]
    if not os.path.exists(_args.sample_dir) or not os.path.exists(_args.hr_dir):
        print('Sample or high-resolution image path doesnt exist')
        return
    if not os.path.exists(_args.plot_dir):
        os.makedirs(_args.plot_dir)
    psnres = {}
    for alg in interpolation_algorithms:
        alg_psnr, alg_time = calculate_sample_avg_psnr(_args.sample_dir, _args.hr_dir, _args.factor, alg[1])
        psnres[alg[0]] = alg_psnr
        print_validation_results(alg[0], alg_psnr, alg_time)
    images = os.listdir(_args.sample_dir)
    interp_count = int(_args.sample_ratio * len(images))
    random_sample = random.sample(images, interp_count)
    for image in random_sample:
        img_path = os.path.join(_args.sample_dir, image)
        out_path = os.path.join(_args.plot_dir, image)
        save_interpolation_result(img_path, out_path, _args.factor, interpolation_algorithms, psnres)
    print(f"Interpolation results are saved in {_args.plot_dir}. Count: {interp_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interpolation")
    parser.add_argument("--sample_dir", type=str, help="Path to sample image directory")
    parser.add_argument("--hr_dir", type=str, help="Path to high-resolution image directory")
    parser.add_argument("--plot_dir", type=str, help="Path to directory where plots will be saved")
    parser.add_argument("--factor", type=float, help="Image upscale factor")
    parser.add_argument("--sample_ratio", type=float, help="Sample image ratio to be plotted")
    args = parser.parse_args()
    main(args)
