import os
import random

import numpy as np
import rasterio
from matplotlib import pyplot as plt
from tqdm import tqdm

from modules import PixelCorrector


def generate_points(img, cnt, k):
    s = set()
    c, h, w = img.shape
    img2 = np.copy(img)
    res = []
    for i in range(cnt):
        x = random.randint(k // 2, h - 1 - k // 2)
        y = random.randint(k // 2, w - 1 - k // 2)
        ch = random.randint(0, c - 1)
        while (x, y) in s:
            x = random.randint(k // 2, h - 1 - k // 2)
            y = random.randint(k // 2, w - 1 - k // 2)
        t = random.randint(0, 1)
        s.add((x, y))
        res.append((ch, x, y))
        if t:
            img2[ch][x][y] = img2[ch][x][y] * 5
        else:
            img2[ch][x][y] = img2[ch][x][y] // 5
    return img2, set(res)


def plt_show(img):
    plt.imshow(np.transpose(img[:-1], axes=[1, 2, 0]) // 255 * 6)
    plt.show()


at_res = 0
af_res = 0
mae = 0
total_cnt = 0
channel_cnt = [0] * 4
k = 5
pad_width = ((0, 0), (k // 2, k // 2), (k // 2, k // 2))
cnt = 20
attempts = 1

p_v = 0
gen_noise = False
data_dir = 'E:\ml_hakaton\data\original\crops'

if gen_noise:
    data_dir = 'E:\ml_hakaton\data\gen_result'

files = os.listdir(data_dir)[:5]

for filename in tqdm(files):
    with rasterio.open(os.path.join(data_dir, filename)) as src:
        original_img = src.read()

    plt_show(original_img)
    t_res = 0
    f_res = 0

    if gen_noise:
        for at in range(attempts):
            im = np.copy(original_img).astype(int)

            noisy_img, noisy_points = generate_points(im, cnt, k)

            plt_show(noisy_img)

            px = PixelCorrector(filename, 'anomalies.csv')
            anomalies = px._correct_rgb(noisy_img)

            # px2 = Pixel2Corrector(filename, 'anomalies2.csv')
            # anomalies = px2._get_data(noisy_img)

            f_cnt = 0
            for p in anomalies:
                p['c'] -= 1
                if (p['c'], p['y'], p['x']) not in noisy_points:
                    f_cnt += 1
                    # print("fake: ", p)
                else:
                    rv = p['correction']
                    value = abs(im[p['c'], p['y'], p['x']] - rv)
                    mae += value
                    p_v += value / im[p['c'], p['y'], p['x']]
                    total_cnt += 1
                    noisy_points.remove((p['c'], p['y'], p['x']))
                    im[p['c'], p['y'], p['x']] = rv

            plt_show(im)

            f_res += f_cnt / attempts

            for p in anomalies:
                # print("not recognized: ", p, noisy_img[p['c'], p['y'], p['x']], original_img[p['c'], p['y'], p['x']])
                channel_cnt[p['c']] += 1

            t_res += (cnt - len(noisy_points)) / attempts

        print('Точность', t_res / cnt)
        print('Среднее количество ложных точек после алгоритма', f_res)

        at_res += t_res / cnt
        af_res += f_res
    else:
        im = np.copy(original_img).astype(int)
        px = PixelCorrector(filename, 'anomalies.csv')
        anomalies = px._correct_rgb(im)
        px._to_csv(anomalies)

        # px2 = Pixel2Corrector(filename, 'anomalies2.csv')
        # anomalies = px2._get_data(im)

        for p in anomalies:
            rv = p['correction']
            p['c'] -= 1
            im[p['c'], p['y'], p['x']] = rv

        plt_show(im)

if gen_noise:
    print('Полная точность', at_res / len(files))
    print('Полное среднее количество ложных точек после алгоритма', af_res / len(files))
    print('Ошибка восстановления', mae / total_cnt, p_v / total_cnt)
    print(channel_cnt)
