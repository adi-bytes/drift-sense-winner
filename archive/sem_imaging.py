"""
SEM acquisition artifacts -- applied per-image, since the reference and
search captures happen under different conditions (careful/slow vs.
fast/wide-area). This is deliberately kept separate from
structural_defects.py, which models a property of the physical device
rather than of how it was imaged.

There is a single physical beam (`beam_spot_size_nm`), applied identically
to both images as a Gaussian PSF blur *before* any downsampling. The search
image's extra softness on dense structures comes naturally from the 10x
area-average downsample on top of that shared blur -- not from a separate
"search-only blur" fudge factor.

Extended from the starter with:
  - Edge brightening (secondary electron emission at topographic edges)
  - Small rotation (stage alignment error)
  - Brightness/contrast jitter (detector gain variation)

Physics references cited in CITATIONS.md.
"""

from __future__ import annotations

import logging
from typing import TypedDict

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class SearchTransform(TypedDict):
    """Geometry applied after downsampling the search image.

    ``row_shift`` is indexed by output row and follows OpenCV remap's
    destination-to-source convention.  The remaining fields describe the
    subsequent radial distortion and rotation stages.
    """

    row_shift: np.ndarray
    barrel_distortion_k: float
    rotation_deg: float
    shape: tuple[int, int]


def transform_search_point(
    point: tuple[float, float], transform: SearchTransform
) -> tuple[float, float]:
    """Map a point from the undistorted search image into output coordinates.

    This mirrors the actual image operations rather than applying a guessed
    forward formula.  In particular, ``apply_raster_drift`` uses a remap whose
    map is output -> source, so the point displacement is the negative row
    shift.  Radial distortion is inverted numerically because the image code
    also uses an output -> source remap.
    """
    x, y = map(float, point)
    h, w = transform["shape"]
    row_shift = transform["row_shift"]
    row = int(np.clip(round(y), 0, h - 1))
    x -= float(row_shift[row])

    k = float(transform["barrel_distortion_k"])
    if k != 0.0:
        cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
        # Solve map(output) == source with a few fixed-point iterations.
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
# Beam-Spot PSF
# ---------------------------------------------------------------------------


def gaussian_psf_blur(
    img: np.ndarray,
    spot_size_nm: float,
    pixel_size_nm: float,
    astigmatism_ratio: float = 1.0,
) -> np.ndarray:
    """Gaussian beam-spot blur. `astigmatism_ratio` != 1.0 makes the spot
    elliptical (sigmaY = sigmaX * ratio) -- a real, common SEM aberration
    where the beam isn't perfectly round, which shows up as directional
    blurring (sharper along one scan axis than the other).

    The FWHM-to-sigma relation is: sigma = FWHM / (2 * sqrt(2 * ln(2))) ≈ FWHM / 2.355.
    Here we use spot_size_nm directly as a sigma-like parameter (already
    calibrated to the starter's conventions) rather than dividing by 2.355,
    keeping backward compatibility with the starter's preset values.
    """
    sigma_x = max(spot_size_nm / pixel_size_nm, 1e-6)
    sigma_y = max(sigma_x * astigmatism_ratio, 1e-6)
    k = int(2 * round(3 * max(sigma_x, sigma_y)) + 1)
    k = max(k, 3)
    return cv2.GaussianBlur(img, (k, k), sigmaX=sigma_x, sigmaY=sigma_y)


# ---------------------------------------------------------------------------
# Edge Brightening (secondary electron emission)
# ---------------------------------------------------------------------------


def apply_edge_brightening(
    img: np.ndarray,
    gain: float,
) -> np.ndarray:
    """Simulate increased secondary electron (SE) emission at topographic
    edges. Real SEM images show bright edges because more SEs escape from
    inclined/edge surfaces than from flat regions (Reimer, Ch. 4).

    Implementation: compute Sobel edge magnitude, normalize to [0, 1],
    and add ``gain * edge_magnitude * 255`` to the image. Applied *before*
    noise since this is a physical SE yield effect, not an imaging artifact.

    Args:
        img: Grayscale uint8 image (the rendered pattern canvas).
        gain: Brightness boost scale factor. 0.0 = no effect; 0.3 = typical.
    """
    if gain <= 0:
        return img
    img_f = img.astype(np.float64)
    # Sobel edge detection — magnitude of x and y gradients
    grad_x = cv2.Sobel(img_f, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img_f, cv2.CV_64F, 0, 1, ksize=3)
    edge_mag = np.sqrt(grad_x**2 + grad_y**2)
    # Normalize edge magnitude to [0, 1]
    max_mag = edge_mag.max()
    if max_mag > 0:
        edge_mag /= max_mag
    out = img_f + gain * edge_mag * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Vignetting
# ---------------------------------------------------------------------------


def apply_vignette(img: np.ndarray, strength: float) -> np.ndarray:
    """Radial darkening toward the frame edges, from off-axis beam/detector
    collection efficiency falloff. `strength` in [0, 1]; 0 = no effect.
    """
    if strength <= 0:
        return img
    h, w = img.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    r = np.sqrt(((yy - cy) / cy) ** 2 + ((xx - cx) / cx) ** 2)
    r = np.clip(r / np.sqrt(2), 0, 1)
    falloff = 1.0 - strength * (r**2)
    out = img.astype(np.float64) * falloff
    return np.clip(out, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Gamma / Nonlinear Contrast
# ---------------------------------------------------------------------------


def apply_gamma(img: np.ndarray, gamma: float) -> np.ndarray:
    """Nonlinear contrast/brightness response curve (detector gain nonlinearity
    or contrast/brightness knob mis-calibration). gamma=1.0 is a no-op.
    """
    if gamma == 1.0:
        return img
    norm = img.astype(np.float64) / 255.0
    out = np.power(np.clip(norm, 0, 1), gamma) * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Barrel / Pincushion Distortion
# ---------------------------------------------------------------------------


def apply_barrel_distortion(img: np.ndarray, k: float) -> np.ndarray:
    """Radial lens-style distortion (barrel if k>0, pincushion if k<0) from
    imperfect beam-scan linearity/calibration. k=0.0 is a no-op.
    """
    if k == 0.0:
        return img
    h, w = img.shape
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    nx = (xx - cx) / cx
    ny = (yy - cy) / cy
    r2 = nx**2 + ny**2
    factor = 1.0 + k * r2
    map_x = (nx * factor) * cx + cx
    map_y = (ny * factor) * cy + cy
    return cv2.remap(
        img,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


# ---------------------------------------------------------------------------
# Rotation (stage alignment error)
# ---------------------------------------------------------------------------


def apply_rotation(img: np.ndarray, angle_deg: float) -> np.ndarray:
    """Small rotation around the image center simulating stage misalignment.
    angle_deg = 0.0 is a no-op. Uses border replication to avoid black edges.
    """
    if angle_deg == 0.0:
        return img
    h, w = img.shape
    center = (w / 2.0, h / 2.0)
    mat = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    return cv2.warpAffine(
        img,
        mat,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


# ---------------------------------------------------------------------------
# Brightness / Contrast Jitter
# ---------------------------------------------------------------------------


def apply_brightness_contrast_jitter(
    img: np.ndarray,
    brightness_jitter: float,
    contrast_jitter: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Random per-image brightness and contrast shift simulating detector
    gain and offset variation between captures.

    Args:
        brightness_jitter: Max absolute brightness shift (in pixel values).
        contrast_jitter: Max fractional contrast scale change (e.g. 0.1 = ±10%).
    """
    if brightness_jitter <= 0 and contrast_jitter <= 0:
        return img
    img_f = img.astype(np.float64)
    if contrast_jitter > 0:
        scale = 1.0 + rng.uniform(-contrast_jitter, contrast_jitter)
        mean = img_f.mean()
        img_f = (img_f - mean) * scale + mean
    if brightness_jitter > 0:
        offset = rng.uniform(-brightness_jitter, brightness_jitter)
        img_f += offset
    return np.clip(img_f, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Charging Streaks
# ---------------------------------------------------------------------------


def add_charging_streaks(
    img: np.ndarray,
    streak_prob: float,
    intensity: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Occasional bright horizontal streaks from local sample charging
    (common on insulating/oxide regions under e-beam). `streak_prob` is the
    expected streaks per 100 rows; `intensity` scales streak brightness.
    """
    if streak_prob <= 0 or intensity <= 0:
        return img
    h, w = img.shape
    out = img.astype(np.float64)
    expected = streak_prob * (h / 100.0)
    n_streaks = rng.poisson(max(expected, 0))
    for _ in range(n_streaks):
        row = int(rng.integers(0, h))
        band = max(1, int(rng.normal(2, 1)))
        lo, hi = max(row - band, 0), min(row + band, h)
        out[lo:hi, :] += intensity * rng.uniform(0.5, 1.0) * 255.0 / 10.0
    return np.clip(out, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Downsample
# ---------------------------------------------------------------------------


def downsample_area_average(img: np.ndarray, factor: int) -> np.ndarray:
    """Area-average downsample by an integer factor."""
    h, w = img.shape
    return cv2.resize(img, (w // factor, h // factor), interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------------------
# Raster Scan Drift
# ---------------------------------------------------------------------------


def apply_raster_drift(
    img: np.ndarray,
    shear_amplitude_px: float,
    jitter_std_px: float,
    rng: np.random.Generator | None,
    row_shift: np.ndarray | None = None,
) -> np.ndarray:
    """Progressive row-to-row shear (drift accumulating over scan time) plus
    per-row jitter (vibration), mimicking real raster-scan drift artifacts.
    """
    if shear_amplitude_px == 0 and jitter_std_px == 0:
        return img
    h, w = img.shape
    rows = np.arange(h)
    shear = shear_amplitude_px * (rows / max(h - 1, 1))
    if row_shift is None:
        if rng is None:
            raise ValueError("rng is required when row_shift is not supplied")
        jitter = (
            rng.normal(0, jitter_std_px, size=h)
            if jitter_std_px > 0
            else np.zeros(h)
        )
        row_shift = (shear + jitter).astype(np.float32)
    elif len(row_shift) != h:
        raise ValueError("row_shift must have one value per image row")

    map_x = np.arange(w, dtype=np.float32)[None, :] + row_shift[:, None]
    map_y = np.tile(np.arange(h, dtype=np.float32)[:, None], (1, w))
    return cv2.remap(
        img,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


# ---------------------------------------------------------------------------
# Shot Noise (Poisson)
# ---------------------------------------------------------------------------


def add_shot_noise(
    img: np.ndarray, dose: float, rng: np.random.Generator
) -> np.ndarray:
    """Poisson shot noise. `dose` is a proxy for electron count/dwell time --
    higher dose (slower/careful scan) means less relative noise.
    Variance equals mean signal: I_noisy ~ Poisson(I * dose / 255) * 255 / dose.
    """
    img_f = img.astype(np.float64)
    counts = np.clip(img_f / 255.0 * dose, 0, None)
    noisy_counts = rng.poisson(counts).astype(np.float64)
    noisy = noisy_counts / dose * 255.0
    return np.clip(noisy, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Detector / Readout Noise (additive Gaussian)
# ---------------------------------------------------------------------------


def add_detector_noise(
    img: np.ndarray, sigma: float, rng: np.random.Generator
) -> np.ndarray:
    """Additive Gaussian noise modeling detector electronics readout noise."""
    if sigma <= 0:
        return img
    noisy = img.astype(np.float64) + rng.normal(0, sigma, size=img.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Speckle Noise (multiplicative)
# ---------------------------------------------------------------------------


def add_speckle_noise(
    img: np.ndarray, sigma: float, rng: np.random.Generator
) -> np.ndarray:
    """Multiplicative noise: out = img * (1 + N(0, sigma)). Distinct from
    the additive Gaussian detector noise above -- a stand-in for detector
    gain variation / coherent-interference-style artifacts, where noise
    magnitude scales with signal brightness rather than being constant.
    """
    if sigma <= 0:
        return img
    img_f = img.astype(np.float64)
    noise = rng.normal(0, sigma, size=img.shape)
    out = img_f * (1.0 + noise)
    return np.clip(out, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Salt-and-Pepper (impulse noise)
# ---------------------------------------------------------------------------


def add_salt_and_pepper_noise(
    img: np.ndarray,
    prob: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Impulse noise: a fraction `prob` of pixels are forced to 0 or 255 --
    a stand-in for dead/hot detector pixels or sudden discharge events,
    structurally different from the smooth noise models above.
    """
    if prob <= 0:
        return img
    out = img.copy()
    hit = rng.random(img.shape) < prob
    salt = rng.random(img.shape) < 0.5
    out[hit & salt] = 255
    out[hit & ~salt] = 0
    return out


# ---------------------------------------------------------------------------
# Full imaging pipelines
# ---------------------------------------------------------------------------


def image_reference(
    crop: np.ndarray,
    pixel_size_nm: float,
    spot_size_nm: float,
    dose: float,
    rng: np.random.Generator,
    detector_noise_sigma: float = 2.0,
    drift_jitter_px: float = 0.2,
    astigmatism_ratio: float = 1.0,
    vignette_strength: float = 0.0,
    gamma: float = 1.0,
    barrel_distortion_k: float = 0.0,
    charging_streak_prob: float = 0.0,
    charging_streak_intensity: float = 0.0,
    speckle_sigma: float = 0.0,
    salt_pepper_prob: float = 0.0,
    edge_brightness_gain: float = 0.0,
    brightness_jitter: float = 0.0,
    contrast_jitter: float = 0.0,
) -> np.ndarray:
    """Full reference-image acquisition pipeline.

    Order mirrors real SEM physics:
    1. Edge brightening (physical SE yield — before any imaging artifacts)
    2. PSF blur (beam optics)
    3. Raster drift (scan artifacts — minimal for careful reference)
    4. Barrel distortion (scan nonlinearity)
    5. Shot noise (fundamental Poisson)
    6. Detector noise (electronics readout)
    7. Speckle noise (detector gain variation)
    8. Salt & pepper (dead/hot pixels)
    9. Vignette (off-axis collection)
    10. Gamma (detector nonlinearity)
    11. Charging streaks (sample charging)
    12. Brightness/contrast jitter (global gain offset)
    """
    img = apply_edge_brightening(crop, edge_brightness_gain)
    img = gaussian_psf_blur(img, spot_size_nm, pixel_size_nm, astigmatism_ratio)
    img = apply_raster_drift(
        img, shear_amplitude_px=0.0, jitter_std_px=drift_jitter_px, rng=rng
    )
    img = apply_barrel_distortion(img, barrel_distortion_k)
    img = add_shot_noise(img, dose, rng)
    img = add_detector_noise(img, detector_noise_sigma, rng)
    img = add_speckle_noise(img, speckle_sigma, rng)
    img = add_salt_and_pepper_noise(img, salt_pepper_prob, rng)
    img = apply_vignette(img, vignette_strength)
    img = apply_gamma(img, gamma)
    img = add_charging_streaks(
        img, charging_streak_prob, charging_streak_intensity, rng
    )
    img = apply_brightness_contrast_jitter(img, brightness_jitter, contrast_jitter, rng)
    return img


def image_search(
    full_canvas: np.ndarray,
    pixel_size_ref_nm: float,
    pixel_size_search_nm: float,
    spot_size_nm: float,
    dose: float,
    rng: np.random.Generator,
    shear_amplitude_px: float = 1.5,
    drift_jitter_px: float = 0.5,
    detector_noise_sigma: float = 5.0,
    astigmatism_ratio: float = 1.0,
    vignette_strength: float = 0.0,
    gamma: float = 1.0,
    barrel_distortion_k: float = 0.0,
    charging_streak_prob: float = 0.0,
    charging_streak_intensity: float = 0.0,
    speckle_sigma: float = 0.0,
    salt_pepper_prob: float = 0.0,
    edge_brightness_gain: float = 0.0,
    rotation_deg: float = 0.0,
    brightness_jitter: float = 0.0,
    contrast_jitter: float = 0.0,
    return_transform: bool = False,
) -> np.ndarray | tuple[np.ndarray, SearchTransform]:
    """Full search-image acquisition pipeline.

    Order mirrors real SEM physics:
    1. Edge brightening (physical SE yield — on the fine canvas)
    2. PSF blur (on the fine canvas, before downsample)
    3. Area-average downsample (10x — pixel-size conversion)
    4. Raster drift (on downsampled search image — heavier than reference)
    5. Barrel distortion (scan nonlinearity)
    6. Shot noise (lower dose → noisier)
    7. Detector noise (electronics readout)
    8. Speckle noise
    9. Salt & pepper
    10. Vignette
    11. Gamma
    12. Charging streaks
    13. Rotation (stage misalignment — applied last for geometry)
    14. Brightness/contrast jitter
    """
    factor = int(round(pixel_size_search_nm / pixel_size_ref_nm))
    canvas = apply_edge_brightening(full_canvas, edge_brightness_gain)
    blurred = gaussian_psf_blur(
        canvas, spot_size_nm, pixel_size_ref_nm, astigmatism_ratio
    )
    downsampled = downsample_area_average(blurred, factor)
    row_shift = np.zeros(downsampled.shape[0], dtype=np.float32)
    if shear_amplitude_px != 0 or drift_jitter_px != 0:
        h = downsampled.shape[0]
        rows = np.arange(h)
        shear = shear_amplitude_px * (rows / max(h - 1, 1))
        jitter = (
            rng.normal(0, drift_jitter_px, size=h)
            if drift_jitter_px > 0
            else np.zeros(h)
        )
        row_shift = (shear + jitter).astype(np.float32)
    drifted = apply_raster_drift(
        downsampled, shear_amplitude_px, drift_jitter_px, rng=None, row_shift=row_shift
    )
    distorted = apply_barrel_distortion(drifted, barrel_distortion_k)
    noisy = add_shot_noise(distorted, dose, rng)
    noisy = add_detector_noise(noisy, detector_noise_sigma, rng)
    noisy = add_speckle_noise(noisy, speckle_sigma, rng)
    noisy = add_salt_and_pepper_noise(noisy, salt_pepper_prob, rng)
    noisy = apply_vignette(noisy, vignette_strength)
    noisy = apply_gamma(noisy, gamma)
    noisy = add_charging_streaks(
        noisy, charging_streak_prob, charging_streak_intensity, rng
    )
    noisy = apply_rotation(noisy, rotation_deg)
    noisy = apply_brightness_contrast_jitter(
        noisy, brightness_jitter, contrast_jitter, rng
    )
    if return_transform:
        transform: SearchTransform = {
            "row_shift": row_shift,
            "barrel_distortion_k": barrel_distortion_k,
            "rotation_deg": rotation_deg,
            "shape": noisy.shape,
        }
        return noisy, transform
    return noisy
