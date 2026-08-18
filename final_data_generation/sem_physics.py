"""
Upgraded SEM physics engine for final_data_generation.

Key upgrades over src/sem_imaging.py:
  - Smooth temporal drift trajectory: GP-style random walk via low-pass filtered
    cumulative noise, replacing the single linear shear model.
    REF: Maraghechi et al. Ultramicroscopy 187 (2018) — smooth temporal drift
    is the dominant geometric artifact in raster-scan SEM.

  - Correlated scan-line shifts: a SEPARATE correlated process from drift,
    representing short-range beam-positioning vibration/jitter.
    REF: Maraghechi et al. Mechanics of Materials — scan-line shifts are
    distinct from smooth drift and must be modeled independently.

  - Correlated electronic noise: FFT-based spatial filtering of white noise
    to produce a realistic noise power spectral density (PSD), replacing
    the white Gaussian detector noise model.
    REF: Villarrubia et al. SPIE 5038 (2003) — CD-SEM noise PSD is NOT white;
    it has a spatial correlation structure that must be modeled correctly.

  - Detector response model: replaces gamma with a physically motivated
    gain + offset + saturation + optional nonlinearity model.
    REF: Li et al. Scanning 35 (2013) — detector gain/offset variation
    is a primary source of CD-SEM measurement variability.

  - Deprioritized: speckle and salt-and-pepper are now minor optional effects,
    not primary noise sources.

Preserved from src/sem_imaging.py:
  - PSF blur, astigmatism, edge brightening, vignetting
  - Barrel distortion, charging streaks, rotation
  - Poisson shot noise (physically correct dose-dependent model)
  - Area-average downsampling
"""
from __future__ import annotations

import logging
from typing import TypedDict

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Type alias for search-image geometry transform (preserved from src/)
# ---------------------------------------------------------------------------

class SearchTransform(TypedDict):
    row_shift: np.ndarray
    barrel_distortion_k: float
    rotation_deg: float
    shape: tuple[int, int]


def transform_search_point(
    point: tuple[float, float], transform: SearchTransform
) -> tuple[float, float]:
    """Map a point from undistorted search coordinates to output coordinates."""
    x, y = map(float, point)
    h, w = transform["shape"]
    row_shift = transform["row_shift"]
    row = int(np.clip(round(y), 0, h - 1))
    x -= float(row_shift[row])

    k = float(transform["barrel_distortion_k"])
    if k != 0.0:
        cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
        qx, qy = x, y
        for _ in range(12):
            nx = (qx - cx) / cx
            ny = (qy - cy) / cy
            factor = 1.0 + k * (nx * nx + ny * ny)
            qx = cx + (x - cx) / max(factor, 1e-6)
            qy = cy + (y - cy) / max(factor, 1e-6)
        x, y = qx, qy

    angle = float(transform["rotation_deg"])
    if angle != 0.0:
        theta = np.radians(angle)
        cx, cy = w / 2.0, h / 2.0
        dx, dy = x - cx, y - cy
        x = cx + dx * np.cos(theta) - dy * np.sin(theta)
        y = cy + dx * np.sin(theta) + dy * np.cos(theta)

    return x, y


# ---------------------------------------------------------------------------
# NEW: Smooth Temporal Drift Trajectory
# REF: Maraghechi et al. Ultramicroscopy 187 (2018).
# Models drift as a smooth random walk: cumulative integral of low-pass
# filtered white noise. The correlation_rows parameter controls how
# slowly the drift changes (longer = smoother, lower frequency content).
# ---------------------------------------------------------------------------

def generate_drift_trajectory(
    n_rows: int,
    amplitude_px: float,
    correlation_rows: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate a smooth temporal drift trajectory for raster acquisition.

    Returns an array of shape (n_rows,) with per-row x-displacement in px.
    The trajectory is a cumulative random walk with Gaussian-correlated
    increments, producing smooth low-frequency drift without abrupt jumps.
    """
    if amplitude_px <= 0 or n_rows == 0:
        return np.zeros(n_rows, dtype=np.float32)

    # Generate white noise increments, smooth them, accumulate
    increments = rng.normal(0, amplitude_px / max(n_rows ** 0.5, 1), n_rows)
    smooth_increments = gaussian_filter1d(increments, sigma=max(correlation_rows, 1.0))

    trajectory = np.cumsum(smooth_increments)
    # Zero-mean: drift can go both ways equally
    trajectory -= trajectory.mean()
    # Rescale so RMS matches the requested amplitude
    rms = trajectory.std()
    if rms > 1e-6:
        trajectory = trajectory / rms * amplitude_px * 0.5
    return trajectory.astype(np.float32)


# ---------------------------------------------------------------------------
# NEW: Correlated Scan-Line Shifts (separate from smooth drift)
# REF: Maraghechi et al. Mechanics of Materials — scan-line shifts are a
# DISTINCT artifact from smooth drift. They represent short-range beam
# positioning errors (vibration, flyback noise) at each raster line.
# ---------------------------------------------------------------------------

def generate_scanline_shifts(
    n_rows: int,
    sigma_px: float,
    correlation_rows: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate short-range correlated scan-line shifts.

    These are short-range (~few rows) correlated displacements, distinct from
    the smooth long-range temporal drift trajectory.
    """
    if sigma_px <= 0 or n_rows == 0:
        return np.zeros(n_rows, dtype=np.float32)
    white = rng.normal(0, sigma_px, n_rows)
    return gaussian_filter1d(white, sigma=max(correlation_rows, 1.0)).astype(np.float32)


# ---------------------------------------------------------------------------
# UPGRADED: Apply Combined Drift + Scan-Line Shifts via remap
# The final row shift = smooth_drift + scanline_shift
# ---------------------------------------------------------------------------

def apply_combined_drift(
    img: np.ndarray,
    drift_trajectory: np.ndarray,
    scanline_shifts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply combined smooth drift + scan-line shifts via OpenCV remap.

    Returns (remapped_image, total_row_shift) where total_row_shift is
    stored in the SearchTransform for ground-truth coordinate correction.
    """
    h, w = img.shape
    row_shift = (drift_trajectory + scanline_shifts).astype(np.float32)
    map_x = np.arange(w, dtype=np.float32)[None, :] + row_shift[:, None]
    map_y = np.tile(np.arange(h, dtype=np.float32)[:, None], (1, w))
    remapped = cv2.remap(
        img, map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return remapped, row_shift


# ---------------------------------------------------------------------------
# UPGRADED: Correlated Electronic/Detector Noise
# REF: Villarrubia et al. SPIE 5038 (2003) — CD-SEM noise PSD is not white;
# it has a spatial correlation structure. We model this by low-pass filtering
# white Gaussian noise to match a realistic noise PSD shape.
# ---------------------------------------------------------------------------

def add_correlated_noise(
    img: np.ndarray,
    sigma: float,
    correlation_length_px: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Add spatially correlated Gaussian noise with a realistic PSD.

    Generates white Gaussian noise then applies a 2D Gaussian spatial filter
    to produce correlation_length_px spatial coherence. The result has
    more power at low spatial frequencies, matching real detector noise PSD.
    """
    if sigma <= 0:
        return img
    white = rng.normal(0, sigma, img.shape)
    if correlation_length_px > 1.0:
        # 2D Gaussian blur of the noise field to produce spatial correlation
        k = max(3, int(2 * round(3 * correlation_length_px) + 1))
        if k % 2 == 0:
            k += 1
        correlated = cv2.GaussianBlur(
            white.astype(np.float32), (k, k), sigmaX=correlation_length_px
        )
        # Re-normalize to preserve sigma after smoothing
        std = correlated.std()
        if std > 1e-8:
            correlated = correlated / std * sigma
        noise = correlated
    else:
        noise = white.astype(np.float32)

    out = img.astype(np.float64) + noise
    return np.clip(out, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# UPGRADED: Detector Response Model (replaces gamma-as-primary-artifact)
# REF: Li et al. Scanning 35 (2013) — gain/offset/saturation variation.
# Gamma is kept as a minor nonlinearity option, NOT a primary SEM effect.
# ---------------------------------------------------------------------------

def apply_detector_response(
    img: np.ndarray,
    gain: float = 1.0,
    offset: float = 0.0,
    nonlinearity: float = 1.0,
    saturation: int = 255,
) -> np.ndarray:
    """Detector gain + offset + optional nonlinearity + ADC saturation.

    I_out = clip(gain * I_in^nonlinearity + offset, 0, saturation)

    This replaces 'gamma' as the primary nonlinear response mechanism,
    reframing it as a detector electronics artifact rather than an SEM
    physics effect.
    """
    out = img.astype(np.float64)
    if nonlinearity != 1.0:
        norm = np.clip(out / 255.0, 0, 1)
        out = np.power(norm, nonlinearity) * 255.0
    out = gain * out + offset
    return np.clip(out, 0, saturation).astype(np.uint8)


# ---------------------------------------------------------------------------
# Preserved from src/sem_imaging.py (verbatim, no changes needed)
# ---------------------------------------------------------------------------

def gaussian_psf_blur(
    img: np.ndarray,
    spot_size_nm: float,
    pixel_size_nm: float,
    astigmatism_ratio: float = 1.0,
) -> np.ndarray:
    """Gaussian beam-spot blur with astigmatism support."""
    sigma_x = max(spot_size_nm / pixel_size_nm, 1e-6)
    sigma_y = max(sigma_x * astigmatism_ratio, 1e-6)
    k = int(2 * round(3 * max(sigma_x, sigma_y)) + 1)
    k = max(k, 3)
    return cv2.GaussianBlur(img, (k, k), sigmaX=sigma_x, sigmaY=sigma_y)


def apply_edge_brightening(img: np.ndarray, gain: float) -> np.ndarray:
    """Secondary electron edge brightening (topographic SE yield increase)."""
    if gain <= 0:
        return img
    img_f = img.astype(np.float64)
    grad_x = cv2.Sobel(img_f, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img_f, cv2.CV_64F, 0, 1, ksize=3)
    edge_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
    max_mag = edge_mag.max()
    if max_mag > 0:
        edge_mag /= max_mag
    out = img_f + gain * edge_mag * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_vignette(img: np.ndarray, strength: float) -> np.ndarray:
    """Radial darkening toward frame edges from off-axis collection falloff."""
    if strength <= 0:
        return img
    h, w = img.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    r = np.sqrt(((yy - cy) / cy) ** 2 + ((xx - cx) / cx) ** 2)
    r = np.clip(r / np.sqrt(2), 0, 1)
    falloff = 1.0 - strength * (r ** 2)
    return np.clip(img.astype(np.float64) * falloff, 0, 255).astype(np.uint8)


def apply_barrel_distortion(img: np.ndarray, k: float) -> np.ndarray:
    """Radial lens distortion (barrel k>0, pincushion k<0)."""
    if k == 0.0:
        return img
    h, w = img.shape
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    nx = (xx - cx) / cx
    ny = (yy - cy) / cy
    r2 = nx ** 2 + ny ** 2
    factor = 1.0 + k * r2
    map_x = (nx * factor) * cx + cx
    map_y = (ny * factor) * cy + cy
    return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def apply_rotation(img: np.ndarray, angle_deg: float) -> np.ndarray:
    """Small rotation simulating stage misalignment."""
    if angle_deg == 0.0:
        return img
    h, w = img.shape
    mat = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle_deg, 1.0)
    return cv2.warpAffine(img, mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def add_shot_noise(img: np.ndarray, dose: float, rng: np.random.Generator) -> np.ndarray:
    """Poisson shot noise. Higher dose = lower relative noise."""
    img_f = img.astype(np.float64)
    counts = np.clip(img_f / 255.0 * dose, 0, None)
    noisy_counts = rng.poisson(counts).astype(np.float64)
    return np.clip(noisy_counts / dose * 255.0, 0, 255).astype(np.uint8)


def add_speckle_noise(img: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """Multiplicative noise (minor optional robustness augmentation only)."""
    if sigma <= 0:
        return img
    img_f = img.astype(np.float64)
    noise = rng.normal(0, sigma, size=img.shape)
    return np.clip(img_f * (1.0 + noise), 0, 255).astype(np.uint8)


def add_salt_and_pepper_noise(img: np.ndarray, prob: float, rng: np.random.Generator) -> np.ndarray:
    """Rare impulse noise (dead/hot pixels). Keep probability very low."""
    if prob <= 0:
        return img
    out = img.copy()
    hit = rng.random(img.shape) < prob
    salt = rng.random(img.shape) < 0.5
    out[hit & salt] = 255
    out[hit & ~salt] = 0
    return out


def add_charging_streaks(
    img: np.ndarray, streak_prob: float, intensity: float, rng: np.random.Generator
) -> np.ndarray:
    """Horizontal charging streaks (insulator charging artifact)."""
    if streak_prob <= 0 or intensity <= 0:
        return img
    h, _w = img.shape
    out = img.astype(np.float64)
    n_streaks = rng.poisson(max(streak_prob * (h / 100.0), 0))
    for _ in range(n_streaks):
        row = int(rng.integers(0, h))
        band = max(1, int(rng.normal(2, 1)))
        lo, hi = max(row - band, 0), min(row + band, h)
        out[lo:hi, :] += intensity * rng.uniform(0.5, 1.0) * 255.0 / 10.0
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_brightness_contrast_jitter(
    img: np.ndarray, brightness_jitter: float, contrast_jitter: float, rng: np.random.Generator
) -> np.ndarray:
    """Per-image brightness/contrast variation (detector gain offset)."""
    if brightness_jitter <= 0 and contrast_jitter <= 0:
        return img
    img_f = img.astype(np.float64)
    if contrast_jitter > 0:
        scale = 1.0 + rng.uniform(-contrast_jitter, contrast_jitter)
        mean = img_f.mean()
        img_f = (img_f - mean) * scale + mean
    if brightness_jitter > 0:
        img_f += rng.uniform(-brightness_jitter, brightness_jitter)
    return np.clip(img_f, 0, 255).astype(np.uint8)


def downsample_area_average(img: np.ndarray, factor: int) -> np.ndarray:
    """Area-average downsample by an integer factor (proper anti-aliasing)."""
    h, w = img.shape
    return cv2.resize(img, (w // factor, h // factor), interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------------------
# Upgraded Full Acquisition Pipelines
# ---------------------------------------------------------------------------

def image_reference(
    crop: np.ndarray,
    pixel_size_nm: float,
    spot_size_nm: float,
    dose: float,
    rng: np.random.Generator,
    correlated_noise_sigma: float = 2.0,
    correlated_noise_length_px: float = 1.0,
    drift_amplitude_px: float = 0.0,
    drift_correlation_rows: float = 50.0,
    scanline_shift_sigma_px: float = 0.0,
    scanline_shift_correlation: float = 5.0,
    astigmatism_ratio: float = 1.0,
    vignette_strength: float = 0.0,
    barrel_distortion_k: float = 0.0,
    charging_streak_prob: float = 0.0,
    charging_streak_intensity: float = 0.0,
    speckle_sigma: float = 0.0,
    salt_pepper_prob: float = 0.0,
    edge_brightness_gain: float = 0.3,
    detector_gain: float = 1.0,
    detector_offset: float = 0.0,
    detector_nonlinearity: float = 1.0,
    brightness_jitter: float = 0.0,
    contrast_jitter: float = 0.0,
) -> np.ndarray:
    """Reference acquisition pipeline with upgraded physics.

    Reference uses lower drift (careful scan) and lower noise (higher dose).
    Order mirrors real SEM physics:
    1. Edge brightening (SE yield — physical, before imaging)
    2. PSF blur (beam optics)
    3. Smooth drift + scan-line shifts (minimal for reference)
    4. Barrel distortion
    5. Poisson shot noise (dose-dependent)
    6. Correlated electronic noise (replaces white Gaussian)
    7. Speckle (minor robustness augmentation)
    8. Salt & pepper (very rare)
    9. Vignette
    10. Detector response (gain/offset/nonlinearity)
    11. Charging streaks
    12. Brightness/contrast jitter
    """
    img = apply_edge_brightening(crop, edge_brightness_gain)
    img = gaussian_psf_blur(img, spot_size_nm, pixel_size_nm, astigmatism_ratio)

    # Reference: minimal drift (0.2x of search drift amplitude)
    drift = generate_drift_trajectory(img.shape[0], drift_amplitude_px * 0.2, drift_correlation_rows, rng)
    shifts = generate_scanline_shifts(img.shape[0], scanline_shift_sigma_px * 0.2, scanline_shift_correlation, rng)
    img, _ = apply_combined_drift(img, drift, shifts)

    img = apply_barrel_distortion(img, barrel_distortion_k * 0.3)
    img = add_shot_noise(img, dose, rng)
    img = add_correlated_noise(img, correlated_noise_sigma * 0.4, correlated_noise_length_px, rng)
    img = add_speckle_noise(img, speckle_sigma, rng)
    img = add_salt_and_pepper_noise(img, salt_pepper_prob, rng)
    img = apply_vignette(img, vignette_strength * 0.5)
    img = apply_detector_response(img, detector_gain, detector_offset, detector_nonlinearity)
    img = add_charging_streaks(img, charging_streak_prob, charging_streak_intensity, rng)
    img = apply_brightness_contrast_jitter(img, brightness_jitter * 0.3, contrast_jitter * 0.3, rng)
    return img


def image_search(
    full_canvas: np.ndarray,
    pixel_size_ref_nm: float,
    pixel_size_search_nm: float,
    spot_size_nm: float,
    dose: float,
    rng: np.random.Generator,
    correlated_noise_sigma: float = 6.0,
    correlated_noise_length_px: float = 2.0,
    drift_amplitude_px: float = 1.5,
    drift_correlation_rows: float = 50.0,
    scanline_shift_sigma_px: float = 0.5,
    scanline_shift_correlation: float = 5.0,
    astigmatism_ratio: float = 1.0,
    vignette_strength: float = 0.0,
    barrel_distortion_k: float = 0.0,
    charging_streak_prob: float = 0.0,
    charging_streak_intensity: float = 0.0,
    speckle_sigma: float = 0.0,
    salt_pepper_prob: float = 0.0,
    edge_brightness_gain: float = 0.3,
    rotation_deg: float = 0.0,
    detector_gain: float = 1.0,
    detector_offset: float = 0.0,
    detector_nonlinearity: float = 1.0,
    brightness_jitter: float = 0.0,
    contrast_jitter: float = 0.0,
    return_transform: bool = False,
) -> np.ndarray | tuple[np.ndarray, SearchTransform]:
    """Search acquisition pipeline with upgraded physics.

    Search uses full drift + scan-line shifts and lower dose (noisier).
    Order mirrors real SEM physics:
    1. Edge brightening (on fine canvas)
    2. PSF blur (on fine canvas, before downsample)
    3. Area-average downsample 10x (pixel-size conversion with anti-aliasing)
    4. Smooth temporal drift trajectory (heavier than reference)
    5. Correlated scan-line shifts (independent from drift)
    6. Barrel distortion
    7. Poisson shot noise (lower dose → noisier)
    8. Correlated electronic noise
    9. Speckle (minor robustness only)
    10. Salt & pepper (rare)
    11. Vignette
    12. Detector response (gain/offset/nonlinearity)
    13. Charging streaks
    14. Rotation (stage misalignment)
    15. Brightness/contrast jitter
    """
    factor = round(pixel_size_search_nm / pixel_size_ref_nm)
    canvas_e = apply_edge_brightening(full_canvas, edge_brightness_gain)
    blurred = gaussian_psf_blur(canvas_e, spot_size_nm, pixel_size_ref_nm, astigmatism_ratio)
    downsampled = downsample_area_average(blurred, factor)

    # Search: full smooth drift + independent scan-line shifts
    drift = generate_drift_trajectory(
        downsampled.shape[0], drift_amplitude_px, drift_correlation_rows, rng
    )
    shifts = generate_scanline_shifts(
        downsampled.shape[0], scanline_shift_sigma_px, scanline_shift_correlation, rng
    )
    drifted, row_shift = apply_combined_drift(downsampled, drift, shifts)

    distorted = apply_barrel_distortion(drifted, barrel_distortion_k)
    noisy = add_shot_noise(distorted, dose, rng)
    noisy = add_correlated_noise(noisy, correlated_noise_sigma, correlated_noise_length_px, rng)
    noisy = add_speckle_noise(noisy, speckle_sigma, rng)
    noisy = add_salt_and_pepper_noise(noisy, salt_pepper_prob, rng)
    noisy = apply_vignette(noisy, vignette_strength)
    noisy = apply_detector_response(noisy, detector_gain, detector_offset, detector_nonlinearity)
    noisy = add_charging_streaks(noisy, charging_streak_prob, charging_streak_intensity, rng)
    noisy = apply_rotation(noisy, rotation_deg)
    noisy = apply_brightness_contrast_jitter(noisy, brightness_jitter, contrast_jitter, rng)

    if return_transform:
        transform: SearchTransform = {
            "row_shift": row_shift,
            "barrel_distortion_k": barrel_distortion_k,
            "rotation_deg": rotation_deg,
            "shape": noisy.shape,
        }
        return noisy, transform
    return noisy
