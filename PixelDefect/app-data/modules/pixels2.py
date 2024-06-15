import numpy as np
import pandas as pd
import rasterio
from constants import RESULT_FOLDER

class Pixel2Corrector:

    def __init__(self, crop_path, result_filename, k=3):
        self.crop_path = crop_path
        self.result_filename = f"{RESULT_FOLDER}/{result_filename}"
        self.k = k

    def _restore_pixel(self, image, center_x, center_y):
        # Размер окрестности
        neighborhood_size = 7
        half_size = neighborhood_size // 2

        # Определяем границы окрестности
        xmin = center_x - half_size
        xmax = center_x + half_size + 1
        ymin = center_y - half_size
        ymax = center_y + half_size + 1

        # Получаем окрестность
        neighborhood = image[xmin:xmax, ymin:ymax]

        restored_value = np.median(neighborhood)

        return restored_value

    def _get_delta_across_channel(self, img, i, j, ch, k):
        mean_delta = 0
        ini = i + k // 2
        inj = j + k // 2
        c, h, w = img.shape

        for ch1 in range(c):
            if ch1 != ch:
                for i1 in range(k):
                    for j1 in range(k):
                        if i1 != k // 2 or j1 != k // 2:
                            mean_delta += abs(img[ch1][ini][inj] - img[ch1][i + i1][j + j1])

        mean_delta = mean_delta / (k * k - 1) / (c - 1)

        return mean_delta

    def _get_anomaly_points(self, img):
        res = []
        c_cnt = 0
        c, h, w = img.shape

        for i in range(h - self.k):
            for j in range(w - self.k):
                med_ar = []
                for ch in range(c):
                    med_ar.append(np.median(img[ch, i: i + self.k, j: j + self.k]))

                ini = i + self.k // 2
                inj = j + self.k // 2
                for ch in range(c):
                    pv = img[ch, ini, inj]
                    th = med_ar[ch]
                    if pv > 5 * th:
                        res.append((ch, ini, inj, 1))
                    else:
                        if pv < th / 5:
                            res.append((ch, ini, inj, 0))
                        else:
                            mean_delta = self._get_delta_across_channel(img, i, j, ch, self.k)
                            if ch < 3 and abs(pv - th) > 10 * mean_delta:
                                c_cnt += 1
                                res.append((ch, ini, inj, 0 if pv < th else 1))
                            else:
                                if ch == 3 and abs(pv - th) > 30 * mean_delta:
                                    c_cnt += 1
                                    res.append((ch, ini, inj, 0 if pv < th else 1))

        # print('Канальная корреляция', c_cnt)

        return res

    def _get_data(self, original_img):

        pad_width = ((0, 0), (self.k // 2, self.k // 2), (self.k // 2, self.k // 2))

        original_img = np.pad(original_img, pad_width, mode='reflect')

        points = self._get_anomaly_points(original_img)
        print('Количество точек аномалий', (len(points)))

        anomalies = []
        for p in points:
            corrected_value = self._restore_pixel(original_img[p[0]], p[1], p[2])
            anomalies.append({
                'y': p[1],
                'x': p[2],
                'c': p[0],
                'value': original_img[p[0], p[1], p[2]],
                'correction': int(corrected_value)
            })

        return anomalies

    def correct(self):
        with rasterio.open(self.crop_path) as src:
            original_img = src.read()

        anomalies = self._get_data(original_img)
        self._to_csv(anomalies)
        return

    def _to_csv(self, anomalies):
        df = pd.DataFrame(anomalies)
        df.to_csv(self.result_filename)
        return
