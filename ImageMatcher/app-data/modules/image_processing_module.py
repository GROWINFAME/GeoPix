import numpy as np
import cv2
import rasterio
from skimage import exposure
from skimage.morphology import disk, square
from skimage.filters import rank
from PIL import Image, ImageOps

class ImageProcessor:
    def __init__(self,
                 layout_path,
                 scene_path,
                 resolution_layout=10,
                 resolution_scene=70):
        self.resolution_scene = resolution_scene
        self.resolution_base = resolution_layout
        self.reduction_factor = resolution_layout / resolution_scene
        self.clahe = cv2.createCLAHE(clipLimit=15.0, tileGridSize=(8, 8))

        self.layout, self.layout_read, self.layout_all_ch, self.layout_img, self.layout_nir = self.load_image(layout_path)
        self.scene, self.scene_read, self.scene_all_ch, self.scene_img, self.scene_nir = self.load_image(scene_path)

    def load_image(self, file_path):
        image = rasterio.open(file_path)
        image_read = image.read([1, 2, 3, 4])
        image_all_ch = np.moveaxis(image_read, 0, -1)
        image_rgb = image_all_ch[:, :, 0:3]
        image_nir = image_all_ch[:, :, -1]
        return image, image_read, image_all_ch, image_rgb, image_nir

    def load_preprocessing(self):
        self.scene_rgb = self.apply_preprocessing(image=self.scene_img, image_type='scene', color_type='rgb')
        self.layout_rgb = self.apply_preprocessing(image=self.layout_img, image_type='layout', color_type='rgb')

        self.scene_gray = self.apply_preprocessing(image=self.scene_img, image_type='scene', color_type='grayscale')
        self.layout_gray = self.apply_preprocessing(image=self.layout_img, image_type='layout', color_type='grayscale')

        self.scene_nir = self.apply_preprocessing(image=self.scene_nir, image_type='scene', color_type='nir')
        self.layout_nir = self.apply_preprocessing(image=self.layout_nir, image_type='layout', color_type='nir')

        self.scene_ndvi = compute_ndvi(self.scene_nir, self.scene_rgb[:, :, 0])
        self.layout_ndvi = compute_ndvi(self.layout_nir, self.layout_rgb[:, :, 0])

    def apply_preprocessing(self, image, image_type, color_type):
        if(image_type == 'layout'):
            return self.layout_preprocessing(image, color_type)
        if(image_type == 'scene'):
            return self.scene_preprocessing(image, color_type)
    def layout_preprocessing(self, image, color_type):
        if(color_type=='rgb'):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if(color_type=='grayscale'):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if color_type != 'nir':
            image = cv2.GaussianBlur(image, (7, 7), 0)
            image = resize_image(image, self.reduction_factor)
        elif color_type == 'nir':
            image = resize_image(image, self.reduction_factor)

        if(len(image.shape) == 2) and color_type == 'gray':
            image = self.clahe.apply(image) # с CLAHE больше features, но поменьше inliers

        if color_type == 'rgb':
            channels = [image[..., i] for i in range(image.shape[-1])]
            equalized_channels = [exposure.equalize_hist(channel) for channel in channels]
            image = np.stack(equalized_channels, axis=-1)
        else:
            image = exposure.equalize_hist(image)
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        return image

    def scene_preprocessing(self, image, color_type):
        if (color_type == 'rgb'):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if (color_type == 'grayscale'):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if (len(image.shape) == 2) and color_type == 'gray':
            image = self.clahe.apply(image)  # с CLAHE больше features, но поменьше inliers

        if color_type == 'rgb':
            channels = [image[..., i] for i in range(image.shape[-1])]
            equalized_channels = [exposure.equalize_hist(channel) for channel in channels]
            image = np.stack(equalized_channels, axis=-1)
        else:
            image = exposure.equalize_hist(image)
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        return image

    def load_edge_processing(self, cloud_resistance=True):
        scene_gradients = compute_gradients(self.scene_gray) + compute_gradients(self.scene_nir) + compute_gradients(self.scene_ndvi)
        layout_gradients = compute_gradients(self.layout_gray) + compute_gradients(self.layout_nir) + compute_gradients(self.layout_ndvi)
        scene_edges = exposure.equalize_hist(scene_gradients)
        layout_edges = exposure.equalize_hist(layout_gradients)
        scene_edges = cv2.normalize(scene_edges, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        layout_edges = cv2.normalize(layout_edges, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        if cloud_resistance:
            _, binary_mask = cv2.threshold(self.layout_rgb[:, :, 2], 220, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            min_area = 500
            mask_for_equalization = np.zeros_like(binary_mask)
            for contour in contours:
                if cv2.contourArea(contour) >= min_area:
                    cv2.drawContours(mask_for_equalization, [contour], -1, 255, thickness=cv2.FILLED)

            layout_edges_new = cv2.normalize(layout_gradients, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            layout_edges_new = rank.equalize(layout_edges_new, footprint=square(95))
            layout_edges[mask_for_equalization == 255] = layout_edges_new[mask_for_equalization == 255]

        self.scene_edges = np.asarray(ImageOps.autocontrast(Image.fromarray(scene_edges), cutoff=10))
        self.layout_edges = np.asarray(ImageOps.autocontrast(Image.fromarray(layout_edges), cutoff=20))  # 20 defualt


def resize_image(base_image, reduction_factor):
    # Получаем новые размеры изображения
    new_width = int(base_image.shape[1] * reduction_factor)
    new_height = int(base_image.shape[0] * reduction_factor)
    # Уменьшаем размер изображения
    resized_image = cv2.resize(base_image, (new_width, new_height))
    return resized_image


# Normalized Difference Vegetation Index (NDVI): используется для оценки здоровья растительности.
def compute_ndvi(nir, red):
    nir = nir.astype(float)
    red = red.astype(float)
    ndvi = (nir - red) / (nir + red + 1e-10)
    ndvi = np.clip(ndvi, -1, 1)
    ndvi = cv2.normalize(ndvi, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)  # Нормализация в диапазон 0-255
    return ndvi

# Enhanced Vegetation Index (EVI): используется для улучшения чувствительности к плотной растительности и уменьшения атмосферных эффектов.
def compute_evi(nir, red, blue, G=2.5, C1=6, C2=7.5, L=1):
    nir = nir.astype(float)
    red = red.astype(float)
    blue = blue.astype(float)
    evi = G * ((nir - red) / (nir + C1 * red - C2 * blue + L + 1e-10))
    evi = np.clip(evi, -1, 1)
    evi = cv2.normalize(evi, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U) # Нормализация в диапазон 0-255
    return evi

# Visible Atmospherically Resistant Index (VARI) : используется для оценки зелености растительности с учетом атмосферных эффектов.
def compute_vari(green, red, blue):
    green = green.astype(float)
    red = red.astype(float)
    blue = blue.astype(float)
    vari = (green - red) / (green + red - blue + 1e-10)
    vari = np.clip(vari, -1, 1)
    vari = cv2.normalize(vari, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return vari

def compute_gradients(image, kernel=3):
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=kernel)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=kernel)
    image = cv2.magnitude(grad_x, grad_y)
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    return image