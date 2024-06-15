from submission_module import SubmissionModule
from image_matcher_module import ImageMatcher
from image_processing_module import ImageProcessor
import numpy as np
from rasterio.crs import CRS
from datetime import datetime
import os

class MainModule:
    def __init__(self,
                 output_save_dir):
        self.submission_module = SubmissionModule(output_path=output_save_dir)
        self.img_matcher = ImageMatcher()
        self.img_matcher.setup_parameters(method='SIFT',
                                          corner_detector='FAST',
                                          distance_metric='L2',
                                          homography_estim_method='RANSAC',
                                          matcher_type='FLANN',
                                          affine_resistance=True,
                                          affine_resistance_scene_only=True,
                                          ratio_test=0.75,
                                          draw_matches_on_img=False)

    def make_matching(self, scene_path, layout_path, refinement=True):

        scene_name = os.path.basename(scene_path)
        layout_name = os.path.basename(layout_path)

        image_processing_module = ImageProcessor(layout_path=layout_path,
                                                 scene_path=scene_path,
                                                 resolution_layout=10,
                                                 resolution_scene=70)

        start_time = datetime.now()

        image_processing_module.load_preprocessing()
        image_processing_module.load_edge_processing(cloud_resistance=True)

        scene_edges = image_processing_module.scene_edges
        layout_edges = image_processing_module.layout_edges

        dst, num_inliers = self.img_matcher.matching(scene_img=scene_edges.astype(np.uint8),
                                                     layout_img=layout_edges.astype(np.uint8))
        if refinement:
            if num_inliers < 300:
                new_dst, new_num_inliers = self.img_matcher.refine_detection(dst=dst,
                                                                         offset=200,
                                                                         scene_img=scene_edges.astype(np.uint8),
                                                                         layout_img=layout_edges.astype(np.uint8))
                if new_num_inliers >= num_inliers + 10:
                    dst = new_dst

        spatial_coords = self.submission_module.transform_coordinates_to_epgs(dst=dst / (image_processing_module.reduction_factor),
                                                                              layout=image_processing_module.layout,
                                                                              target_crs=CRS.from_epsg(32637))

        result = self.submission_module.write_results_to_csv_file(file_path='final_coords.csv',
                                                         scene_name=scene_name,
                                                         layout_name=layout_name,
                                                         bounds=spatial_coords,
                                                         crs=CRS.from_epsg(32637),
                                                         start_time=start_time,
                                                         end_time=datetime.now())
        
        return result



