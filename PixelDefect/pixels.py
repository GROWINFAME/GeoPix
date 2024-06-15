import numpy as np
import pandas as pd
import rasterio

from skimage.exposure import equalize_hist


class PixelCorrector:

    def __init__(self, crop_path, result_filename, P=3, Q=3, threshold=99.9):
        self.crop_path = crop_path
        self.result_filename = result_filename
        self.P = P
        self.Q = Q
        self.threshold = threshold

    def _read_file(self):
        with rasterio.open(self.crop_path) as f:
            img = f.read([1, 2, 3])
        return img

    def _get_g(self, i, j, c):
        _c = c[i - self.P // 2:i + self.P // 2 + 1, j - self.Q // 2:j + self.Q // 2 + 1]
        lambda_c = np.ones((self.P, self.Q), dtype=float) * c[i, j]
        g_c = lambda_c - _c
        return g_c

    def _get_cd_hat(self, i, j, c, g_d):
        lambda_cd_hat = c[i - self.P // 2:i + self.P // 2 + 1, j - self.Q // 2:j + self.Q // 2 + 1] + g_d
        lambda_cd_hat[self.P // 2, self.Q // 2] = 0
        return lambda_cd_hat

    def _get_c_hat(self, i, j, c, g_1, g_2):
        lambda_c1_hat = self._get_cd_hat(i, j, c, g_1)
        lambda_c2_hat = self._get_cd_hat(i, j, c, g_2)
        lambda_c_hat = np.mean((lambda_c1_hat + lambda_c2_hat) / 2)
        if lambda_c_hat > 1:
            lambda_c_hat = 1
        if lambda_c_hat < 0:
            lambda_c_hat = 0
        return lambda_c_hat

    def _get_correction(self, img, c, i, j):
        area = img[c, i - self.P // 2:i + self.P // 2 + 1, j - self.Q // 2:j + self.Q // 2 + 1].copy()
        area[self.P // 2, self.Q // 2] = 0
        return int(area.sum() / (self.P * self.Q - 1))

    def _correct_rgb(self, img):
        rgb = equalize_hist(np.moveaxis(img, 0, -1))

        r = rgb[:, :, 0]
        g = rgb[:, :, 1]
        b = rgb[:, :, 2]

        pad_width = ((self.P // 2, self.P // 2), (self.Q // 2, self.Q // 2))

        r_padded = np.pad(r, pad_width, mode='reflect')
        g_padded = np.pad(g, pad_width, mode='reflect')
        b_padded = np.pad(b, pad_width, mode='reflect')

        results_r = np.zeros_like(r)
        results_g = np.zeros_like(g)
        results_b = np.zeros_like(b)

        for i in range(self.P // 2, r_padded.shape[0] - self.P // 2):
            for j in range(self.Q // 2, r_padded.shape[1] - self.Q // 2):
                g_r = self._get_g(i, j, r_padded)
                g_g = self._get_g(i, j, g_padded)
                g_b = self._get_g(i, j, b_padded)

                lambda_b_hat = self._get_c_hat(i, j, b_padded, g_r, g_g)
                lambda_g_hat = self._get_c_hat(i, j, g_padded, g_r, g_b)
                lambda_r_hat = self._get_c_hat(i, j, r_padded, g_g, g_b)

                results_r[i - self.P // 2, j - self.Q // 2] = lambda_r_hat
                results_g[i - self.P // 2, j - self.Q // 2] = lambda_g_hat
                results_b[i - self.P // 2, j - self.Q // 2] = lambda_b_hat

        diff_r = np.abs(r - results_r)
        diff_g = np.abs(g - results_g)
        diff_b = np.abs(b - results_b)

        R_max = np.percentile(diff_r.flatten(), self.threshold)
        R_min = np.percentile(diff_r.flatten(), 100 - self.threshold)

        G_max = np.percentile(diff_g.flatten(), self.threshold)
        G_min = np.percentile(diff_g.flatten(), 100 - self.threshold)

        B_max = np.percentile(diff_b.flatten(), self.threshold)
        B_min = np.percentile(diff_b.flatten(), 100 - self.threshold)

        anomalies = []

        for i in range(self.P // 2, img.shape[1] - self.P // 2 + 1):
            for j in range(self.Q // 2, img.shape[2] - self.Q // 2 + 1):
                if (diff_r[i, j] > R_max) | (diff_r[i, j] < R_min):
                    anomalies.append({
                        'y': i,
                        'x': j,
                        'c': 1,
                        'value': img[0, i, j],
                        'correction': self._get_correction(img, 0, i, j)
                    })
                elif (diff_g[i, j] > G_max) | (diff_g[i, j] < G_min):
                    anomalies.append({
                        'y': i,
                        'x': j,
                        'c': 2,
                        'value': img[1, i, j],
                        'correction': self._get_correction(img, 1, i, j)
                    })
                elif (diff_b[i, j] > B_max) | (diff_b[i, j] < B_min):
                    anomalies.append({
                        'y': i,
                        'x': j,
                        'c': 3,
                        'value': img[2, i, j],
                        'correction': self._get_correction(img, 2, i, j)
                    })

        return anomalies

    def _to_csv(self, anomalies):
        df = pd.DataFrame(anomalies)
        df.to_csv(self.result_filename)
        return

    def correct(self):
        img = self._read_file()
        anomalies = self._correct_rgb(img)
        self._to_csv(anomalies)
        return
