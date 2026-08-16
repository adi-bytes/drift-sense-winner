"""Boundary-signature verification for candidate SEM patches."""

from __future__ import annotations

import cv2
import numpy as np


def extract_boundary_signature(img: np.ndarray, border_px: int = 5) -> np.ndarray:
    """Return concatenated top, bottom, left, and right border pixels."""
    h, w = img.shape
    if border_px <= 0 or h < 2 * border_px or w < 2 * border_px:
        return np.array([], dtype=np.float32)
    return np.concatenate((
        img[:border_px, :].ravel(),
        img[-border_px:, :].ravel(),
        img[:, :border_px].ravel(),
        img[:, -border_px:].ravel(),
    )).astype(np.float32)


def boundary_correlation(ref: np.ndarray, patch: np.ndarray, border_px: int = 5) -> float:
    """Compute Pearson correlation between two image boundary signatures."""
    ref_sig = extract_boundary_signature(ref, border_px)
    patch_sig = extract_boundary_signature(patch, border_px)
    if ref_sig.size == 0 or ref_sig.size != patch_sig.size:
        return -1.0
    ref_std = float(ref_sig.std())
    patch_std = float(patch_sig.std())
    if ref_std < 1e-8 or patch_std < 1e-8:
        return 0.0
    return float(np.corrcoef(ref_sig, patch_sig)[0, 1])


def boundary_score_with_rotation(
    ref: np.ndarray,
    patch: np.ndarray,
    angles: tuple[float, ...] = (-2.0, -1.0, 0.0, 1.0, 2.0),
    border_px: int = 5,
) -> tuple[float, float]:
    """Return the best boundary correlation over small residual rotations."""
    best_score, best_angle = -1.0, 0.0
    h, w = ref.shape
    for angle in angles:
        if angle == 0.0:
            rotated = ref
        else:
            matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
            rotated = cv2.warpAffine(ref, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        score = boundary_correlation(rotated, patch, border_px)
        if score > best_score:
            best_score, best_angle = score, angle
    return float(best_score), float(best_angle)
