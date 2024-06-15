import cv2
import matplotlib.pyplot as plt
import numpy as np
from asift_new import affine_detect, affine_detect_with_corner_detector
import multiprocessing
from multiprocessing.pool import ThreadPool

class ImageMatcher:
  def __init__(self):
    self.MIN_MATCHES = 4 #  Минимальное количество хороших совпадений, необходимое для успешного сопоставления.

  def setup_parameters(self, method,
                       corner_detector,
                       distance_metric,
                       homography_estim_method,
                       matcher_type,
                       affine_resistance=False,
                       affine_resistance_scene_only=False,
                       ratio_test=0.75,
                       draw_matches_on_img=False):
    self.method_name = method
    self.matcher_type = matcher_type
    self.corner_detector = corner_detector
    self.affine_resistance = affine_resistance
    self.affine_resistance_scene_only = affine_resistance_scene_only
    self.ratio_test = ratio_test
    self.draw_matches_on_img = draw_matches_on_img
    self.load_method(method)
    self.load_corner_detector(corner_detector)
    self.load_distance_metric(distance_metric)
    self.load_homography_estim_method(homography_estim_method)

  def load_method(self, method):
    if method == 'AKAZE':
      self.method = cv2.AKAZE.create()
    elif method == 'KAZE':
      self.method = cv2.KAZE.create()
    elif method == 'SIFT':
      self.method = cv2.SIFT_create()

  def load_corner_detector(self, corner_detector):
    if corner_detector == 'FAST':
      self.corner_detector = cv2.FastFeatureDetector_create()
    if corner_detector == 'BRISK':
      self.corner_detector = cv2.BRISK_create()

  def load_distance_metric(self, distance_metric):
    if distance_metric == 'HAMMING':
      self.matcher = cv2.NORM_HAMMING
    elif distance_metric == 'L2':
      self.matcher = cv2.NORM_L2
    elif distance_metric == 'L1':
      self.matcher = cv2.NORM_L1
    elif distance_metric == 'INF':
      self.matcher = cv2.NORM_INF

  def load_homography_estim_method(self, homography_estim_method):
    if homography_estim_method == 'LMEDS':
      self.homog = cv2.LMEDS
    elif homography_estim_method == 'RANSAC':
      self.homog = cv2.RANSAC

  def get_matches(self, descriptors1, descriptors2):
    if self.matcher_type == 'BF':
      bf = cv2.BFMatcher(self.matcher)
      matches = bf.knnMatch(descriptors1, descriptors2, k=2) #  This function finds the k best matches of number for each descriptor from a query set.
    elif self.matcher_type == 'FLANN':
      if(self.matcher == cv2.NORM_L2):
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
      else:
        FLANN_INDEX_LSH = 5
        index_params = dict(algorithm=FLANN_INDEX_LSH,
                            table_number=6,  # 12
                            key_size=12,  # 20
                            multi_probe_level=1)
      # Converto to float32
      descriptors1 = np.float32(descriptors1)
      descriptors2 = np.float32(descriptors2)
      # Create FLANN object
      FLANN = cv2.FlannBasedMatcher(indexParams = index_params, searchParams=dict(checks=50))
      # Matching descriptor vectors using FLANN Matcher
      matches = FLANN.knnMatch(descriptors1, descriptors2, k=2)
    return matches

  def get_key_points_and_descriptors(self, image, image_mask, image_type):
    if not self.affine_resistance:
      if self.corner_detector:
        key_points = self.corner_detector.detect(image, image_mask)
        key_points, descriptors = self.method.compute(image, key_points)
      else:
        key_points, descriptors = self.method.detectAndCompute(image, image_mask)
    else:
      num_cores = multiprocessing.cpu_count()
      if self.corner_detector:
        if (image_type=='layout' and self.affine_resistance_scene_only):
          key_points = self.corner_detector.detect(image, image_mask)
          key_points, descriptors = self.method.compute(image, key_points)
        else:
          with ThreadPool(processes=num_cores) as pool:
            key_points, descriptors = affine_detect_with_corner_detector(self.corner_detector, self.method, image, pool)
      else:
        if (image_type == 'layout' and self.affine_resistance_scene_only):
          key_points, descriptors = self.method.detectAndCompute(image, image_mask)
        else:
          with ThreadPool(processes=num_cores) as pool:
            key_points, descriptors = affine_detect(self.method, image, pool)

    return key_points, descriptors


  def matching(self, scene_img, layout_img, scene_mask=None, layout_mask=None, use_previous_scene_results=False, use_previous_layout_results=False):
    print('Load Matching Process...')
    if not use_previous_scene_results:
      self.key_points1, self.descriptors1 = self.get_key_points_and_descriptors(scene_img, scene_mask, 'scene')
    if not use_previous_layout_results:
      self.key_points2, self.descriptors2 = self.get_key_points_and_descriptors(layout_img, layout_mask, 'layout')
    print(f"scene_img - {len(self.key_points1)} features, layout_img - {len(self.key_points2)} features")

    matches = self.get_matches(self.descriptors1, self.descriptors2)

    # Ratio test по методу Д.Лоу  для фильтрации хороших совпадений
    # Для каждого совпадения вычисляется отношение расстояний между дескриптором и его ближайшим и вторым по близости соседями.
    # Если отношение меньше, чем заданный ratio_test, совпадение считается "хорошим" и добавляется в список good_matches.
    good_matches = []
    for m, n in matches:
        if m.distance < n.distance * self.ratio_test:
            good_matches.append([m])

    # Draw matches between keypoints
    if(self.draw_matches_on_img):
      matching_res = cv2.drawMatchesKnn(cv2.cvtColor(scene_img, cv2.COLOR_RGB2BGR), self.key_points1, cv2.cvtColor(layout_img, cv2.COLOR_RGB2BGR), self.key_points2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS, matchColor=(0, 0, 255))
      cv2.imwrite('matches_res.jpg', matching_res)

    if len(good_matches) > self.MIN_MATCHES:
      src_points = np.float32([self.key_points1[match[0].queryIdx].pt for match in good_matches]).reshape(-1, 1, 2)
      dst_points = np.float32([self.key_points2[match[0].trainIdx].pt for match in good_matches]).reshape(-1, 1, 2)
      # Оценка гомографии
      h, status = cv2.findHomography(src_points, dst_points, self.homog, ransacReprojThreshold=4.5, maxIters=10000, confidence=0.99)
      if h is not None and h.shape == (3, 3):
        num_inliers = np.sum(status)
        print('%d / %d  inliers/matched' % (num_inliers, len(status)))
        t_h, t_w = scene_img.shape[0:2]
        # Get coordinates
        pts = np.float32([[0, 0], [0, t_h], [t_w, t_h], [t_w, 0]]).reshape(-1, 1, 2)
        dst = np.int32(cv2.perspectiveTransform(pts, h))
        dst = np.clip(dst, a_min=0, a_max=None) # если есть отрицательные координаты, то делаем clip до 0
        return dst, num_inliers
      else:
        print('Homography could not be computed')
        dst = np.array([[0, 0], [0, 0], [0, 0], [0, 0]])
        return dst, 0
    else:
      print('Good matches were not found')
      dst = np.array([[0, 0], [0, 0], [0, 0], [0, 0]])
      return dst, 0


  def sliding_window_matching(self, scene_img, layout_img, window_size, step_size):
    best_match = None
    best_dst = None
    max_inliers = 0

    layout_height, layout_width = layout_img.shape[:2]
    window_height, window_width = window_size

    for y in range(0, layout_height - window_height + 1, step_size):
      for x in range(0, layout_width - window_width + 1, step_size):
        window = layout_img[y:y + window_height, x:x + window_width]
        dst, num_inliers = self.matching(scene_img, window)

        if dst is not None:
          if num_inliers > max_inliers:
            max_inliers = num_inliers
            best_dst = dst + np.array([[x, y]])

    return best_dst, max_inliers


  def refine_detection(self, dst, offset, scene_img, layout_img, scene_mask=None, layout_mask=None):
    print('Load Refine Detection Results...')
    # Определение минимальных и максимальных координат
    x_min, y_min = np.min(dst, axis=0)[0]
    x_max, y_max = np.max(dst, axis=0)[0]

    # Добавление отступа
    x_min = max(0, x_min - offset)
    y_min = max(0, y_min - offset)
    x_max = min(layout_img.shape[1], x_max + offset)
    y_max = min(layout_img.shape[0], y_max + offset)

    # Смещение для новых координат
    total_offset = np.array([x_min, y_min])

    # Вырезка новой области подложки и канала NIR
    new_layout_img = layout_img[int(y_min):int(y_max), int(x_min):int(x_max)]
    if(layout_mask):
      new_layout_mask = layout_mask[int(y_min):int(y_max), int(x_min):int(x_max)]
    else:
      new_layout_mask = None

    # Применение метода get_homography для уточненной области
    new_dst, num_inliers = self.matching(scene_img=scene_img,
                                         layout_img=new_layout_img,
                                         scene_mask=scene_mask,
                                         layout_mask=new_layout_mask,
                                         use_previous_scene_results=True,
                                         use_previous_layout_results=False)
    # Возвращение новых координат с учетом смещения
    return new_dst + total_offset, num_inliers

  def draw_detected_location(self, layout_img, dst):
    cv2.polylines(layout_img, [dst], isClosed=True, color=(255, 0, 0), thickness=2)
    plt.figure(figsize=(15, 15))
    plt.imshow(layout_img, cmap='gray')
    plt.show()
