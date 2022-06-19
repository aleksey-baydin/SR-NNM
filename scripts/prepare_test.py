import argparse
import os
import shutil
import math
import cv2
import numpy as np
from tqdm import tqdm
import config
from imgproc import image_resize


def main(_args) -> None:
    if os.path.exists(_args.gt12_output_dir):
        shutil.rmtree(_args.gt12_output_dir)
    os.makedirs(_args.gt12_output_dir)
    if os.path.exists(_args.lrbic4_output_dir):
        shutil.rmtree(_args.lrbic4_output_dir)
    os.makedirs(_args.lrbic4_output_dir)
    # Получение названий файлов изображений в директории
    image_file_names = os.listdir(_args.test_images_dir)
    # Подготовка тестовой выборки в нескольких потоках
    progress_bar = tqdm(image_file_names, total=len(image_file_names), unit="image", desc="Prepare test images")
    for image_file_name in progress_bar:
        test_worker(image_file_name, _args)


def test_worker(image_file_name, _args) -> None:
    image = cv2.imread(f"{_args.test_images_dir}/{image_file_name}", cv2.IMREAD_UNCHANGED)
    image_height, image_width = image.shape[0:2]
    h_excess, w_excess = image_height % 12, image_width % 12
    ih, iw = image_height - h_excess, image_width - w_excess
    n_pos_y = math.ceil(h_excess // 2)
    n_pos_x = math.ceil(w_excess // 2)
    crop_image = image[n_pos_y: n_pos_y + ih, n_pos_x:n_pos_x + iw, ...]
    crop_image = np.ascontiguousarray(crop_image)
    lrbic4_filepath = f"{_args.lrbic4_output_dir}/{image_file_name}"
    # Сохранение LR-изображения
    cv2.imwrite(lrbic4_filepath, crop_image)
    image = cv2.imread(lrbic4_filepath, cv2.IMREAD_UNCHANGED)
    image = image_resize(image, config.upscale_factor)
    gt12_filepath = f"{_args.gt12_output_dir}/{image_file_name}"
    # Сохранение HR-изображения
    cv2.imwrite(gt12_filepath, image)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare database scripts.")
    parser.add_argument("--test_images_dir", type=str, help="Path to input image directory.")
    parser.add_argument("--gt12_output_dir", type=str, help="Path to gt12 image directory.")
    parser.add_argument("--lrbic4_output_dir", type=str, help="Path to lrbic4 image directory.")
    parser.add_argument("--num_workers", type=int, help="How many threads to open at the same time.")
    args = parser.parse_args()
    main(args)
