import requests

# URL для запроса
url = 'http://localhost:8000/api/'

# Пути к файлам
file_path = '/home/nikita/workprojects/хакатон/GeoPix/PixelDefect/crop_2_2_0000.tif'

# Заголовки запроса
headers = {
    'accept': 'application/json'
}

# Файлы для загрузки
files = {

    'scene': ('crop_2_2_0000.tif', open(file_path, 'rb'), 'image/tiff')
}
data = {
    'layout': 'layout_2021-08-16.tif',
}

# Отправка POST-запроса
response = requests.post(url, headers=headers, data=data, files=files)

# Обработка ответа
print(response.status_code)
print(response.json())
