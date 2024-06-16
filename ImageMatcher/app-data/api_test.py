import requests

# URL для запроса
url = 'http://localhost:8000/api/'

# Пути к файлам
file_path = 'E:\ml_hakaton\data\original\crops\crop_0_1_0000.tif'

# Заголовки запроса
headers = {
    'accept': 'application/json'
}

# Файлы для загрузки
files = {

    'scene': (file_path, open(file_path, 'rb'), 'image/tiff')
}
data = {
    'layout': 'layout_2021-08-16.tif',
}

# Отправка POST-запроса
response = requests.post(url, headers=headers, data=data, files=files)

# Обработка ответа
print(response.status_code)
print(response.json())
