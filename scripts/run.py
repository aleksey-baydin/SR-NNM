import os
import config

# Подготовка выборки
os.system("python ./prepare_sample.py "
          "--all_images_dir ../data/pleural_cells/all "
          "--train_output_dir ../data/pleural_cells/train_sample "
          "--test_output_dir ../data/pleural_cells/test_sample "
          "--train_sample_size 200 "
          "--test_sample_size 40")

# Подготовка изображений обучающей выборки
os.system("python ./prepare_dataset.py "
          "--images_dir ../data/pleural_cells/train_sample "
          f"--output_dir ../data/pleural_cells/{config.exp_name} "
          f"--output_bg_dir ../data/pleural_cells/{config.exp_name}/bg "
          f"--output_else_dir ../data/pleural_cells/{config.exp_name}/else "
          f"--image_size {config.image_size} "
          "--step 96 "
          "--bg_coeff 50 "
          "--num_workers 16")

# Подготовка изображений тестовой выборки
os.system("python ./prepare_test.py "
          "--test_images_dir ../data/pleural_cells/test_sample "
          "--gt12_output_dir ../data/pleural_cells/test/GTmod12 "
          "--lrbic4_output_dir ../data/pleural_cells/test/LRbicx4 "
          "--num_workers 16")

# Разделение на обучающую и валидационную выборки
os.system("python ./split_train_valid_dataset.py "
          f"--all_images_dir ../data/pleural_cells/{config.exp_name}/else "
          f"--train_images_dir ../data/pleural_cells/{config.exp_name}/train "
          f"--valid_images_dir ../data/pleural_cells/{config.exp_name}/valid "
          "--valid_samples_ratio 0.2")
