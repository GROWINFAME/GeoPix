import cv2
import numpy as np
from multiprocessing.pool import ThreadPool

def affine_skew(img, tilt, scale_w=1.0, scale_h=1.0, mask=None):
    """
    affine_skew(tilt, img, scale=1.0, scale_w=1.0, scale_h=1.0, mask=None) -> skew_img, skew_mask, Ai

    Perform affine transformations including tilt, scaling, width scaling, and height scaling.
    tilt adjusts the tilt of the image.
    scale adjusts the overall size of the image.
    scale_w adjusts the width of the image.
    scale_h adjusts the height of the image.
    Ai is an affine transform matrix from skew_img to img.
    """
    h, w = img.shape[:2]

    if mask is None:
        mask = np.zeros((h, w), np.uint8)
        mask[:] = 255

    A = np.float32([[1, 0, 0], [0, 1, 0]])

    # Scale width and height separately
    if scale_w != 1.0 or scale_h != 1.0:
        img = cv2.resize(img, (int(w * scale_w), int(h * scale_h)), interpolation=cv2.INTER_NEAREST)
        A = np.dot(np.float32([[scale_w, 0], [0, scale_h]]), A)

    # Tilt image
    if tilt != 1.0:
        s = 0.8 * np.sqrt(tilt * tilt - 1)
        img = cv2.GaussianBlur(img, (0, 0), sigmaX=s, sigmaY=0.01)
        img = cv2.resize(img, (0, 0), fx=1.0 / tilt, fy=1.0, interpolation=cv2.INTER_NEAREST)
        A[0] /= tilt

    if tilt != 1.0 or scale_w != 1.0 or scale_h != 1.0:
        h, w = img.shape[:2]
        mask = cv2.warpAffine(mask, A, (w, h), flags=cv2.INTER_NEAREST)

    Ai = cv2.invertAffineTransform(A)

    return img, mask, Ai

def affine_detect(detector, img, pool=None):
    """
    affine_detect(detector, img, mask=None, pool=None) -> keypoints, descrs

    Apply a set of affine transformations to the image, detect keypoints and
    reproject them into initial image coordinates.

    ThreadPool object may be passed to speedup the computation. Please use multiprocess pool to bypass GIL limitations.
    """
    params = [(1.0, 1.0, 1.0)]  # Initial params with no transformation

    # Generate tilt, scale, and separate width and height scales
    for tilt in 2 ** (0.5 * np.arange(1, 6)):
            for scale_w in [0.8, 1.0, 1.2, 1.5, 1.8]:
                for scale_h in [0.8, 1.0, 1.2, 1.5, 1.8]:
                    params.append((tilt, scale_w, scale_h))

    def f(p):
        tilt, scale_w, scale_h = p
        timg, tmask, Ai = affine_skew(img, tilt, scale_w, scale_h)
        keypoints, descrs = detector.detectAndCompute(timg, tmask)

        for kp in keypoints:
            x, y = kp.pt
            kp.pt = tuple(np.dot(Ai, (x, y, 1)))

        if descrs is None:
            descrs = []

        return keypoints, descrs

    keypoints, descrs = [], []

    if pool is None:
        for p in params:
            k, d = f(p)
            keypoints.extend(k)
            descrs.extend(d)
    else:
        ires = pool.imap(f, params)
        for i, (k, d) in enumerate(ires):
            print(f"Affine sampling: {i + 1} / {len(params)}\r", end='')
            keypoints.extend(k)
            descrs.extend(d)

    print()

    return keypoints, np.array(descrs)

def affine_detect_with_corner_detector(detector_fast, detector_sift, img, pool=None):
    """
    affine_detect(detector, img, mask=None, pool=None) -> keypoints, descrs

    Apply a set of affine transformations to the image, detect keypoints and
    reproject them into initial image coordinates.

    ThreadPool object may be passed to speedup the computation. Please use multiprocess pool to bypass GIL limitations.
    """
    params = [(1.0, 1.0, 1.0)]  # Initial params with no transformation

    # Generate tilt, scale, and separate width and height scales
    for tilt in 2 ** (0.5 * np.arange(1, 6)):
            for scale_w in [0.8, 1.0, 1.2, 1.5, 1.8]:
                for scale_h in [0.8, 1.0, 1.2, 1.5, 1.8]:
                    params.append((tilt, scale_w, scale_h))

    def f(p):
        tilt, scale_w, scale_h = p
        timg, tmask, Ai = affine_skew(img, tilt, scale_w, scale_h)
        keypoints = detector_fast.detect(timg, tmask)

        # Convert keypoints to SIFT keypoints to compute descriptors
        keypoints, descrs = detector_sift.compute(timg, keypoints)

        for kp in keypoints:
            x, y = kp.pt
            kp.pt = tuple(np.dot(Ai, (x, y, 1)))

        if descrs is None:
            descrs = []

        return keypoints, descrs

    keypoints, descrs = [], []

    if pool is None:
        for p in params:
            k, d = f(p)
            keypoints.extend(k)
            descrs.extend(d)
    else:
        ires = pool.imap(f, params)
        for i, (k, d) in enumerate(ires):
            print(f"Affine sampling: {i + 1} / {len(params)}\r", end='')
            keypoints.extend(k)
            descrs.extend(d)

    print()

    return keypoints, np.array(descrs)