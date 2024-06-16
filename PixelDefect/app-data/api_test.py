import numpy as np
import rasterio
import requests
from matplotlib import pyplot as plt

# URL для запроса
url = 'http://localhost:8001/api/'

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

# Отправка POST-запроса
response = requests.post(url, headers=headers, files=files)

# Обработка ответа
print(response.status_code)
with open('restored.tiff', 'wb') as file:
    file.write(response.content)

with rasterio.open(file_path) as src:
    original_img = src.read()

plt.imshow(np.transpose(original_img[:-1], axes=[1, 2, 0]) // 255 * 6)
plt.show()

with rasterio.open('restored.tiff') as src:
    restored_img = src.read()

plt.imshow(np.transpose(restored_img[:-1], axes=[1, 2, 0]) // 255 * 6)
plt.show()
