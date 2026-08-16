"""Lightweight global orientation estimation for SEM images."""

from __future__ import annotations

import cv2
import numpy as np


def estimate_rotation_fft(img: np.ndarray) -> float:
    """Estimate the dominant image orientation in degrees in ``[0, 180)``."""
    if img.ndim != 2 or min(img.shape) < 8:
        return 0.0
    h, w = img.shape
    window = np.outer(np.hanning(h), np.hanning(w)).astype(np.float32)
    spectrum = np.fft.fftshift(np.abs(np.fft.fft2(img.astype(np.float32) * window)))
    cy, cx = h // 2, w // 2
    max_radius = min(cy, cx)
    energies = np.zeros(180, dtype=np.float64)
    for angle in range(180):
        rad = np.deg2rad(angle)
        radii = np.arange(4, max_radius, 2)
        xs = np.rint(cx + radii * np.cos(rad)).astype(int)
        ys = np.rint(cy + radii * np.sin(rad)).astype(int)
        valid = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
        xs2 = np.rint(cx - radii * np.cos(rad)).astype(int)
        ys2 = np.rint(cy - radii * np.sin(rad)).astype(int)
        valid2 = (xs2 >= 0) & (xs2 < w) & (ys2 >= 0) & (ys2 < h)
        samples = np.concatenate((spectrum[ys[valid], xs[valid]], spectrum[ys2[valid2], xs2[valid2]]))
        energies[angle] = float(samples.mean()) if samples.size else 0.0
    return float(np.argmax(energies))


def estimate_rotation_between(ref: np.ndarray, search: np.ndarray) -> float:
    """Estimate the smallest grid rotation from ``ref`` to ``search``."""
    diff = estimate_rotation_fft(search) - estimate_rotation_fft(ref)
    while diff > 45.0:
        diff -= 90.0
    while diff < -45.0:
        diff += 90.0
    return float(diff)


def rotate_image(img: np.ndarray, angle: float, center: tuple[float, float] | None = None) -> np.ndarray:
    """Rotate a grayscale image with replicated borders."""
    h, w = img.shape
    pivot = center if center is not None else (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(pivot, angle, 1.0)
    return cv2.warpAffine(img, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
