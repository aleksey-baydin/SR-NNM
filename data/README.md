# Использование

### Загрузить датасеты

* Качественные стекла https://drive.google.com/drive/folders/1GeD1qNdUKTCl8AhwjzXxS7dNOZuhhyeA
* Дефектные стекла https://drive.google.com/drive/folders/1i6LHZRjB7f6vs2LwfBui5W5RZgXM5LmW

### Подготовить директории

```text
- data
    - pleural_cells
        - all
            - img1.jpg
            - img2.jpg
            ...
        - train_sample (generated via run.py)
        - test_sample (generated via run.py)
        - SRResNet_v1 (generated via run.py)
        - test (generated via run.py)
```

### Запустить разделение изображений на выборки и создание директорий

```bash
cd <SRGAN-_PROJECT-PATH>/scripts
python run.py
```