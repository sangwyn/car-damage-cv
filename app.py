from flask import Flask, request, jsonify, render_template
import base64
import os
import numpy as np
import torch
import io
from PIL import Image
import uuid
import time
import tempfile

# Import your grade functions
# загрузим изображение через URL
import requests
from PIL import Image
import numpy as np
from io import BytesIO

def download_image(url):
    """
    Скачивает изображение по URL, сохраняет его в формате RGB и возвращает массив NumPy.

    :param url: URL изображения
    :return: Массив NumPy изображения в формате RGB
    """
    try:
        # Скачиваем изображение
        response = requests.get(url)
        response.raise_for_status()  # Проверяем, что запрос успешен

        # Открываем изображение из байтового потока
        image = Image.open(BytesIO(response.content)).convert('RGB')

        # Преобразуем изображение в массив NumPy
        image_array = np.array(image)

        return image_array
    except Exception as e:
        print(f"Ошибка при скачивании или обработке изображения: {e}")
        return None

# подгоним по размерам, чтобы скормить нашей модели

import cv2

def resize_and_pad(image):
        h, w = image.shape[:2]
        max_size = 512

        # Вычисляем новые размеры с сохранением пропорций
        if h > w:
            new_h = max_size
            new_w = int(w * max_size / h)
        else:
            new_w = max_size
            new_h = int(h * max_size / w)

        # Изменяем размер
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

        # Создаем новые изображения с черным фоном
        padded_image = np.zeros((max_size, max_size, 3), dtype=image.dtype)

        # Вычисляем позиции для вставки
        top = (max_size - new_h) // 2
        left = (max_size - new_w) // 2

        # Вставляем изображения в центр
        padded_image[top:top+new_h, left:left+new_w] = image

        return padded_image

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER




# загрузка моделей
import segmentation_models_pytorch as smp
import torch

import torchvision
from torchvision.utils import make_grid

from tqdm import tqdm

import os

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

print(f"GPU: {torch.cuda.is_available()}")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("device: ", device)

damage_model, parts_model = None, None

def get_damage_model(model_path):
    global damage_model
    if damage_model is None:
        damage_model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",    # !!!!!!!!
            in_channels=3,
            classes=9,
            activation=None,
            decoder_channels=(256, 128, 64, 32, 16)
        )

        torch.manual_seed(11)
        damage_model = damage_model.to(device)
        # model_path = os.path.join(WSD, './checkpoints/baseline/epoch_101.pth')
        damage_model.load_state_dict(torch.load(model_path, map_location=torch.device(device))['model_state_dict'])

        damage_model = damage_model.eval()

def get_parts_model(model_path):
    global parts_model
    if parts_model is None:
        parts_model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",    # !!!!!!!!
            in_channels=3,
            classes=13,
            activation=None,
            decoder_channels=(256, 128, 64, 32, 16)
        )

        torch.manual_seed(11)
        parts_model = parts_model.to(device)
        parts_model.load_state_dict(torch.load(model_path, map_location=torch.device(device))['model_state_dict'])

        parts_model = parts_model.eval()


import torch
import torch.nn as nn
import torch.nn.functional as F

def get_damage_part_matrix(damage_model, parts_model, image):

    def prepare_image(image):
        car_image = image.astype(np.float32)/255.0
        return torch.from_numpy(car_image).permute(2, 0, 1).float()

    image_tensor = prepare_image(image)[None, ...].to(device)

    damage_maps = F.softmax(damage_model(image_tensor), dim=1)[0]
    parts_maps = F.softmax(parts_model(image_tensor), dim=1)[0]

    # Извлекаем размеры
    num_damage_maps, H, W = damage_maps.shape
    num_parts_maps, _, _ = parts_maps.shape

    # Инициализируем матрицу результата
    result_matrix = torch.zeros((num_damage_maps, num_parts_maps))

    # Вычисляем значения для каждой ячейки матрицы
    for x in range(num_damage_maps):
        for y in range(num_parts_maps):
            T_damage = 0.5
            T_parts = 0.5

            # Извлекаем карты признаков X и Y
            damage_map = damage_maps[x, :, :]  # (H, W)
            parts_map = parts_maps[y, :, :]    # (H, W)

            damage_map_binary = (damage_map >= T_damage).float()  # Бинаризация damage_map
            parts_map_binary = (parts_map >= T_parts).float()    # Бинаризация parts_map

            # Поэлементное произведение бинаризованных карт
            product = damage_map_binary * parts_map_binary  # (H, W)

            # Сумма произведения
            numerator = product.sum()  # Скаляр

            # Сумма бинаризованной карты parts_map
            denominator = parts_map_binary.sum()  # Скаляр

            # Вычисляем значение для ячейки (X, Y)
            result_matrix[x, y] = numerator / denominator if denominator != 0 else 0.0

    return result_matrix


# а теперь все вместе

def get_damage_laval(image_url, damage_model, parts_model):
    # damage_weights = [[6, 6, 0, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
    #                   [4, 3, 0, 2, 0, 2, 2, 0, 4, 4, 4, 5, 4],
    #                   [1, 0, 0, 2, 0, 1, 1, 0, 2, 1, 1, 3, 1],
    #                   [2, 3, 0, 2, 1, 2, 2, 0, 2, 2, 2, 3, 3],
    #                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #                   [2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    #                   [2, 2, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    #                   [1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    #                   [1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]

    damage_weights = [[60, 90, 80, 0, 80, 40, 60, 80, 100, 80, 60, 70, 60],
                      [48, 72, 64, 0, 64, 32, 48, 64, 80, 64, 48, 56, 48],
                      [18, 27, 24, 0, 24, 12, 18, 24, 30, 24, 18, 21, 18],
                      [54, 81, 72, 0, 72, 36, 54, 72, 90, 72, 54, 63, 54],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [30, 45, 40, 0, 40, 20, 30, 40, 50, 40, 30, 35, 30],
                      [42, 63, 56, 0, 56, 28, 42, 56, 70, 56, 42, 49, 42],
                      [36, 54, 48, 0, 48, 24, 36, 48, 60, 48, 36, 42, 36],
                      [24, 36, 32, 0, 32, 16, 24, 32, 40, 32, 24, 28, 24]]

    damage_weights = np.array(damage_weights)

    image = download_image(image_url)

    # image = resize_and_pad(image) # FIXME

    damage_part_matrix = get_damage_part_matrix(damage_model, parts_model, image)

    laval = damage_part_matrix * damage_weights

    # return int(laval.sum().item() * 3)
    return int(laval.sum().item() / 6)

    # return int(laval.sum().item())

    print(damage_part_matrix.shape, damage_part_matrix.sum())

    chisl = laval.sum().item() * 3
    div = float(np.max(damage_weights) * (damage_part_matrix != 0).sum())

    print(chisl, div)

    score = (chisl / div) * 100
    # score = score // 20 + 1

    return score





@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def process_image():
    global damage_model, parts_model
    data = request.json
    image_url = data.get('image_url', '')
    print(image_url)

    get_damage_model("./model_damage.pth")
    get_parts_model("./model_parts.pth")

    result = get_damage_laval(image_url, damage_model, parts_model)
    print(f'!{result}!')
    if result < 1:
        result = 1
    if result > 5:
        result = 5

    return jsonify({'score': result, 'url': image_url}), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)