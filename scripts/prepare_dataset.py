import argparse
import multiprocessing
import os
import shutil
import numpy as np
import cv2
from tqdm import tqdm


def main(_args) -> None:
    if os.path.exists(_args.output_dir):
        shutil.rmtree(_args.output_dir)
    os.makedirs(_args.output_dir)
    os.makedirs(_args.output_bg_dir)
    os.makedirs(_args.output_else_dir)
    # Получение названий файлов изображений в директории
    image_file_names = os.listdir(_args.images_dir)
    # Подготовка изображений в нескольких потоках
    progress_bar = tqdm(total=len(image_file_names), unit="image", desc="Prepare split image")
    workers_pool = multiprocessing.Pool(_args.num_workers)
    for image_file_name in image_file_names:
        workers_pool.apply_async(center_crop_worker, args=(image_file_name, _args),
                                 callback=lambda arg: progress_bar.update(1))
    workers_pool.close()
    workers_pool.join()
    progress_bar.close()


def center_crop_worker(image_file_name, _args) -> None:
    image = cv2.imread(f"{_args.images_dir}/{image_file_name}", cv2.IMREAD_UNCHANGED)
    image_height, image_width = image.shape[0:2]
    half_dim_diff = abs(image_height - image_width) // 2
    less_dim, height_is_less = (image_height, True) if image_height < image_width else (image_width, False)
    index = 1
    imsize = _args.image_size
    if less_dim >= _args.image_size:
        for pos_y in range(0, less_dim - imsize + 1, _args.step):
            for pos_x in range(0, less_dim - imsize + 1, _args.step):
                n_pos_y = pos_y if height_is_less else pos_y + half_dim_diff
                n_pos_x = pos_x + half_dim_diff if height_is_less else pos_x
                # Кроппинг
                crop_image = image[n_pos_y: n_pos_y + imsize, n_pos_x:n_pos_x + imsize, ...]
                u = [int(u.max()) for u in crop_image[0]]
                w = [int(w.min()) for w in crop_image[-1]]
                is_not_bg = np.abs(np.subtract(u, w)).max() > _args.bg_coeff
                # Сохранение кроппированного изображения
                fn = f"{image_file_name.split('.')[-2]}_{index:04d}.{image_file_name.split('.')[-1]}"
                fp = f"{_args.output_else_dir}/{fn}" if is_not_bg else f"{_args.output_bg_dir}/{fn}"
                cv2.imwrite(fp, crop_image)
                index += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare database scripts.")
    parser.add_argument("--images_dir", type=str, help="Path to input image directory.")
    parser.add_argument("--output_dir", type=str, help="Path to output image directory.")
    parser.add_argument("--output_bg_dir", type=str, help="Path to output background image directory.")
    parser.add_argument("--output_else_dir", type=str, help="Path to other output directory.")
    parser.add_argument("--image_size", type=int, help="Low-resolution image size from raw image.")
    parser.add_argument("--step", type=int, help="Crop image similar to sliding window.")
    parser.add_argument("--bg_coeff", type=int, help="Background pixel coefficient.")
    parser.add_argument("--num_workers", type=int, help="How many threads to open at the same time.")
    args = parser.parse_args()
    main(args)
