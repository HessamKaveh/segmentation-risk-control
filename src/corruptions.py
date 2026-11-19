"""Image corruptions that a robot camera actually meets, applied to the full image.

Each corruption has 3 severities. All randomness is seeded per (instance, corruption, severity), so every
run sees exactly the same corrupted images.
"""
import cv2
import numpy as np

CORRUPTIONS = ["motion_blur", "gaussian_noise", "low_light", "jpeg"]


def motion_blur(img, s, rng):
    k = [7, 15, 25][s - 1]
    kern = np.zeros((k, k), np.float32)
    kern[k // 2] = 1.0 / k
    M = cv2.getRotationMatrix2D((k / 2 - 0.5, k / 2 - 0.5), rng.uniform(0, 180), 1.0)
    kern = cv2.warpAffine(kern, M, (k, k))
    kern /= kern.sum()
    return cv2.filter2D(img, -1, kern)


def gaussian_noise(img, s, rng):
    sd = [0.04, 0.08, 0.14][s - 1] * 255
    return np.clip(img.astype(np.float32) + rng.normal(0, sd, img.shape), 0, 255).astype(np.uint8)


def low_light(img, s, rng):
    # darker exposure (gamma + gain) with the shot/read noise that comes with it, as in a dim workcell
    gain, gamma, sd = [(0.5, 1.5, 0.02), (0.3, 2.0, 0.035), (0.18, 2.5, 0.05)][s - 1]
    x = gain * (img.astype(np.float32) / 255.0) ** gamma
    x = x + rng.normal(0, sd, img.shape)
    return np.clip(x * 255, 0, 255).astype(np.uint8)


def jpeg(img, s, rng):
    q = [20, 10, 5][s - 1]
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)


def apply(img, name, severity, seed):
    if name == "clean":
        return img
    return globals()[name](img, severity, np.random.default_rng(seed))
