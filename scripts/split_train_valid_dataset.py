import argparse
import os
import random
import shutil
from tqdm import tqdm


def main(_args) -> None:
    if not os.path.exists(_args.train_images_dir):
        os.makedirs(_args.train_images_dir)
    if not os.path.exists(_args.valid_images_dir):
        os.makedirs(_args.valid_images_dir)
    all_files = os.listdir(_args.all_images_dir)
    random.shuffle(all_files)
    w = int(len(all_files) * _args.valid_samples_ratio)
    train_files = all_files[w:]
    valid_files = all_files[:w]
    train_process_bar = tqdm(train_files, total=len(train_files), unit="image",
                             desc="Split train/valid dataset (train)")
    valid_process_bar = tqdm(valid_files, total=len(valid_files), unit="image",
                             desc="Split train/valid dataset (valid)")
    for image_file_name in train_process_bar:
        shutil.copyfile(f"{_args.all_images_dir}/{image_file_name}", f"{_args.train_images_dir}/{image_file_name}")
    for image_file_name in valid_process_bar:
        shutil.copyfile(f"{_args.all_images_dir}/{image_file_name}", f"{_args.valid_images_dir}/{image_file_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split train and valid dataset scripts.")
    parser.add_argument("--all_images_dir", type=str, help="Path to all image directory.")
    parser.add_argument("--train_images_dir", type=str, help="Path to train image directory.")
    parser.add_argument("--valid_images_dir", type=str, help="Path to valid image directory.")
    parser.add_argument("--valid_samples_ratio", type=float,
                        help="What percentage of the data is extracted from the training set into the validation set.")
    args = parser.parse_args()
    main(args)
