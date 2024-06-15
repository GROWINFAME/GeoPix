import rasterio
import json
from rasterio.transform import from_bounds
import numpy as np
import os
import pandas as pd

class SubmissionModule:
    def __init__(self,
                 output_path):
        self.output_path = os.path.join(output_path, 'submission')
        os.makedirs(self.output_path, exist_ok=True)

    def transform_coordinates_to_epgs(self, dst, layout, target_crs=None):
        if (np.sum(dst) != 0):
            dst = dst[:, 0, :]
            # Получаем преобразование координат изображения
            transform = layout.transform
            # Определяем исходную координатную систему
            src_crs = layout.crs
            spatial_coords = np.array([list(rasterio.transform.xy(transform, pixel[0], pixel[1])) for pixel in dst])
            if src_crs.to_string() != target_crs.to_string():
                spatial_coords_x, spatial_coords_y = rasterio.warp.transform(src_crs, target_crs, spatial_coords[:, 0], spatial_coords[:, 1])
                spatial_coords = np.column_stack((spatial_coords_x, spatial_coords_y))
            return spatial_coords
        else:
            return dst

    def write_coordinates_to_file(self, file_path, coords):
        with open(os.path.join(self.output_path, file_path), 'w') as file:
            for coord in coords:
                file.write(f"{coord[0]:.3f}; {coord[1]:.3f}\n")

    def write_coordinates_to_geojson(self, file_path, coords, scene_name):
        geojson_data = {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [coords.tolist()]
            },
            "properties":{
                "name": scene_name
            }
        }
        with open(os.path.join(self.output_path, file_path), 'w') as file:
            json.dump(geojson_data, file, indent=4)

    def save_scene_with_new_crs_and_bounds(self, file_path, scene, scene_data, bounds, crs):
        # Преобразование new_bounds в формат (xmin, ymin, xmax, ymax)
        bounds = np.array(bounds).reshape(-1, 2)
        xmin, ymin = np.min(bounds, axis=0)
        xmax, ymax = np.max(bounds, axis=0)

        # Определение новых параметров трансформации на основе новых границ и размеров
        new_transform = from_bounds(xmin, ymin, xmax, ymax, width=scene.width, height=scene.height)

        # Обновление метаданных
        new_meta = scene.meta.copy()
        new_meta.update({
            'crs': crs,
            'transform': new_transform,
        })

        # Создание нового GeoTIFF файла с обновленными метаданными
        with rasterio.open(os.path.join(self.output_path, file_path), 'w', **new_meta) as dst:
            dst.write(scene_data)

    def write_results_to_csv_file(self, file_path, layout_name, scene_name, bounds, crs, start_time, end_time):
        columns = ["layout_name", "scene_name", "ul", "ur", "br", "bl", "crs", "start", "end"]

        # Проверка существования файла и создание DataFrame
        if os.path.exists(os.path.join(self.output_path, file_path)):
            df = pd.read_csv(os.path.join(self.output_path, file_path), encoding='utf-8')
        else:
            df = pd.DataFrame(columns=columns)

        new_row = {
            'layout_name': layout_name,
            'scene_name': scene_name,
            'ul': f"{bounds[0][0]:.3f}, {bounds[0][1]:.3f}",
            'ur': f"{bounds[1][0]:.3f}, {bounds[1][1]:.3f}",
            'br': f"{bounds[2][0]:.3f}, {bounds[2][1]:.3f}",
            'bl': f"{bounds[3][0]:.3f}, {bounds[3][1]:.3f}",
            'crs': crs.to_string(),
            'start': start_time.strftime("%Y-%m-%dT%H:%M:%S"),
            'end': end_time.strftime("%Y-%m-%dT%H:%M:%S")
        }
        df.loc[len(df.index)] = list(new_row.values())
        df.to_csv(os.path.join(self.output_path, file_path), index=False, encoding='utf-8')

        return new_row
