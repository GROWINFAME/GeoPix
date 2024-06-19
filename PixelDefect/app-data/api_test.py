import argparse
import numpy as np
import rasterio
import requests
from matplotlib import pyplot as plt
import os

def main(crop_name):
    # URL для запроса
    url = 'http://localhost:8001/api/'

    # Проверка наличия файла
    if not os.path.isfile(crop_name):
        print(f"Ошибка: файл {crop_name} не найден.")
        return

    # Заголовки запроса
    headers = {
        'accept': 'application/json'
    }

    # Файлы для загрузки
    files = {
        'scene': (os.path.basename(crop_name), open(crop_name, 'rb'), 'image/tiff')
    }

    # Отправка POST-запроса
    response = requests.post(url, headers=headers, files=files)

    # Обработка ответа
    print(response.status_code)
    if response.status_code == 200:
        print("Проверьте папку с результатами!")
    else:
        print("Ошибка получения ответа от сервера:", response.text)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Запуск скрипта с параметром crop_name.')
    parser.add_argument('--crop_name', required=True, help='Полный путь к файлу crop_name')

    args = parser.parse_args()

    main(args.crop_name)
