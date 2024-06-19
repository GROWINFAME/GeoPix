import argparse
import requests
import os


def main(crop_name, layout_name):
    # URL для запроса
    url = 'http://localhost:8000/api/'
    
    # Заголовки запроса
    headers = {
        'accept': 'application/json'
    }

    # Проверка наличия файлов
    if not os.path.isfile(crop_name):
        print(f"Ошибка: файл {crop_name} не найден.")
        return

    if not os.path.isfile(layout_name):
        print(f"Ошибка: файл {layout_name} не найден.")
        return

    # Файлы для загрузки
    files = {
        'scene': (os.path.basename(crop_name), open(crop_name, 'rb'), 'image/tiff')
    }

    # Данные для запроса
    data = {
        'layout': os.path.basename(layout_name),
    }

    # Отправка POST-запроса
    response = requests.post(url, headers=headers, data=data, files=files)

    # Обработка ответа
    if response.status_code == 200:
        print("Запрос выполнен успешно.")
        print("Ответ сервера:", response.json())
    else:
        print(f"Ошибка выполнения запроса: {response.status_code}")
        print("Ответ сервера:", response.text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Запуск скрипта с параметрами crop_name и layout_name.')
    parser.add_argument('--crop_name', required=True, help='Полный путь к файлу crop_name')
    parser.add_argument('--layout_name', required=True, help='Полный путь к файлу layout_name')

    args = parser.parse_args()

    main(args.crop_name, args.layout_name)
