from modules import PixelCorrector
from modules import Pixel2Corrector

px = PixelCorrector('E:\ml_hakaton\data\\noise_result\layout_2021-08-16_213_3930.tif', 'anomalies.csv')
px.correct()

px2 = Pixel2Corrector('E:\ml_hakaton\data\\noise_result\layout_2021-08-16_213_3930.tif', 'anomalies2.csv')
px2.correct()
