# Поиск и исправление битых пикселей

Пример использования:

```
from pixels import PixelCorrector

px = PixelCorrector(r'C:\Users\sveta\Python Projects\sitronics_images\1_20\crop_1_0_0000.tif', 'anomalies.csv')
px.correct()
```

Параметры:

- ```crop_path``` - абсолютный путь до снимка
- ```result_filename``` - csv-файл для сохранения результатов
- ```P``` - окрестности точки для восстановления, Y
- ```Q``` - окрестности точки для восстановления, X
- ```threshold``` - персентиль для определения аномалий, значения 0-100

```
from pixels import Pixel2Corrector

px = Pixel2Corrector(r'C:\Users\sveta\Python Projects\sitronics_images\1_20\crop_1_0_0000.tif', 'anomalies.csv')
px.correct()
```

Параметры:

- ```crop_path``` - абсолютный путь до снимка
- ```result_filename``` - csv-файл для сохранения результатов

