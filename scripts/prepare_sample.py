import os
import shutil
import random
import argparse
from tqdm import tqdm


def main(_args) -> None:
    if os.path.exists(_args.train_output_dir):
        shutil.rmtree(_args.train_output_dir)
    os.makedirs(_args.train_output_dir)
    if os.path.exists(_args.test_output_dir):
        shutil.rmtree(_args.test_output_dir)
    os.makedirs(_args.test_output_dir)
    # Получение названий файлов изображений в директории
    image_file_names = os.listdir(_args.all_images_dir)
    if _args.train_sample_size + _args.test_sample_size > len(image_file_names):
        raise ValueError("Not enough images to create both train and test samples!")
    random.shuffle(image_file_names)
    train_sample_files = image_file_names[:_args.train_sample_size]
    test_sample_files = image_file_names[-_args.test_sample_size:]
    # Разделение на выборки в нескольких потоках
    train_pb = tqdm(train_sample_files, total=len(train_sample_files), unit="image", desc="Prepare train sample")
    test_pb = tqdm(test_sample_files, total=len(test_sample_files), unit="image", desc="Prepare test sample")
    for sample in train_pb:
        shutil.copyfile(f"{_args.all_images_dir}/{sample}", f"{_args.train_output_dir}/{sample}")
    for sample in test_pb:
        shutil.copyfile(f"{_args.all_images_dir}/{sample}", f"{_args.test_output_dir}/{sample}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare database scripts.")
    parser.add_argument("--all_images_dir", type=str, help="Path to all image directory.")
    parser.add_argument("--train_output_dir", type=str, help="Path to train sample image directory.")
    parser.add_argument("--test_output_dir", type=str, help="Path to test sample image directory.")
    parser.add_argument("--train_sample_size", type=int, help="Number of images in train sample.")
    parser.add_argument("--test_sample_size", type=int, help="Number of images in test sample.")
    args = parser.parse_args()
    main(args)
