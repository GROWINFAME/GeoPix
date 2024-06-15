import numpy as np
import cv2

def affine_skew(tilt, phi, img, mask=None):
    """
    affine_skew(tilt, phi, img, mask=None) -> skew_img, skew_mask, Ai

    phi is in degrees

    Ai is an affine transform matrix from skew_img to img
    """
    h, w = img.shape[:2]

    if mask is None:
        mask = np.zeros((h, w), np.uint8)
        mask[:] = 255

    A = np.float32([[1, 0, 0], [0, 1, 0]])

    # Rotate image
    if phi != 0.0:
        phi = np.deg2rad(phi)
        s, c = np.sin(phi), np.cos(phi)
        A = np.float32([[c, -s], [s, c]])
        corners = [[0, 0], [w, 0], [w, h], [0, h]]
        tcorners = np.int32(np.dot(corners, A.T))
        x, y, w, h = cv2.boundingRect(tcorners.reshape(1, -1, 2))
        A = np.hstack([A, [[-x], [-y]]])
        img = cv2.warpAffine(img, A, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    # Tilt image (resizing after rotation)
    if tilt != 1.0:
        s = 0.8 * np.sqrt(tilt * tilt - 1)
        img = cv2.GaussianBlur(img, (0, 0), sigmaX=s, sigmaY=0.01)
        img = cv2.resize(img, (0, 0), fx=1.0 / tilt, fy=1.0, interpolation=cv2.INTER_NEAREST)
        A[0] /= tilt

    if phi != 0.0 or tilt != 1.0:
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
    params = [(1.0, 0.0)]

    # Simulate all possible affine transformations
    for t in 2 ** (0.5 * np.arange(1, 6)):
        for phi in np.arange(0, 180, 72.0 / t):
            params.append((t, phi))

    def f(p):
        t, phi = p
        timg, tmask, Ai = affine_skew(t, phi, img)
        keypoints, descrs = detector.detectAndCompute(timg, tmask)

        for kp in keypoints:
            x, y = kp.pt
            kp.pt = tuple(np.dot(Ai, (x, y, 1)))

        if descrs is None:
            descrs = []

        return keypoints, descrs

    keypoints, descrs = [], []

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
    params = [(1.0, 0.0)]

    # Simulate all possible affine transformations
    for t in 2 ** (0.5 * np.arange(1, 6)):
        for phi in np.arange(0, 180, 72.0 / t):
            params.append((t, phi))

    def f(p):
        t, phi = p
        timg, tmask, Ai = affine_skew(t, phi, img)
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

    ires = pool.imap(f, params)

    for i, (k, d) in enumerate(ires):
        print(f"Affine sampling: {i + 1} / {len(params)}\r", end='')
        keypoints.extend(k)
        descrs.extend(d)

    print()

    return keypoints, np.array(descrs)