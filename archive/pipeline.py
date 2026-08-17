"""
Orchestrates one Drift-Sense sample:

  fine canvas (1 nm/px, 10000x10000)
    -> random 1000x1000 crop = Reference Image (native res)
    -> whole-canvas beam blur + 10x downsample + search-specific noise/drift
       = Search Image (1000x1000 @ 10 nm/px)
    -> ground truth = crop location, converted to search-image pixel coords

Physical calibration is fixed by the problem statement: both images are
1000x1000 px; reference is 1 nm/px (1 um FOV), search is 10 nm/px (10 um
FOV). The 10x relationship falls directly out of that pixel-size ratio --
no separate "shrink by 10x" resize step is needed.

Extended from the starter with:
  - edge_brightness_gain: secondary electron edge brightening
  - rotation_deg: small stage alignment rotation on search image
  - brightness_jitter / contrast_jitter: per-image detector gain variation
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

import numpy as np

from src import sem_imaging
from src.patterns.dram import generate_dram_canvas
from src.patterns.finfet import generate_finfet_canvas
from src.patterns.zones import generate_zone_canvas
from src.presets import get_preset

logger = logging.getLogger(__name__)

REFERENCE_SIZE_PX: int = 1000
PIXEL_SIZE_REF_NM: int = 1
PIXEL_SIZE_SEARCH_NM: int = 10
SCALE_FACTOR: int = PIXEL_SIZE_SEARCH_NM // PIXEL_SIZE_REF_NM  # 10
FINE_CANVAS_SIZE_PX: int = REFERENCE_SIZE_PX * SCALE_FACTOR  # 10000


@dataclass
class GenerationParams:
    """All tuneable knobs for synthetic sample generation.

    Each field maps to a physical SEM imaging effect or device structure
    parameter — see CITATIONS.md for literature references.
    """

    beam_spot_size_nm: float = 5.0
    collapse_threshold_nm: float = 10.0
    dose_reference: float = 2000.0
    dose_search: float = 200.0
    shear_amplitude_px: float = 1.5
    drift_jitter_px: float = 0.5
    detector_noise_sigma_ref: float = 2.0
    detector_noise_sigma_search: float = 5.0

    # Astigmatism: beam-spot ellipticity (1.0 = round spot, no effect)
    astigmatism_ratio: float = 1.0
    # Vignetting: radial darkening toward frame edges (0 = none)
    vignette_strength: float = 0.0
    # Nonlinear contrast/gamma response (1.0 = no effect)
    gamma: float = 1.0
    # Barrel(+)/pincushion(-) geometric lens distortion (0 = none)
    barrel_distortion_k: float = 0.0
    # Charging-streak artifacts
    charging_streak_prob: float = 0.0
    charging_streak_intensity: float = 0.0
    # Multiplicative (speckle-style) noise
    speckle_sigma: float = 0.0
    # Impulse (salt-and-pepper) noise
    salt_pepper_prob: float = 0.0

    # Zone composition
    mat_size_nm: float = 2600.0
    strip_width_nm: float = 320.0
    boundary_bias: float = 0.35

    # CD/etch bias and corner rounding
    linewidth_bias_nm: float = 0.0
    corner_rounding_px: float = 0.0

    # --- NEW: Extended physics (not in starter) ---
    # Secondary electron edge brightening gain (0 = none, 0.3 = typical)
    edge_brightness_gain: float = 0.3
    # Search-image rotation in degrees (stage alignment error)
    rotation_deg: float = 0.0
    # Per-image brightness/contrast jitter
    brightness_jitter: float = 0.0
    contrast_jitter: float = 0.0

    def as_dict(self) -> dict:
        """Serialize all params to a flat dict (for manifest CSV)."""
        return asdict(self)


_GENERATORS: dict = {
    "dram": generate_dram_canvas,
    "finfet": generate_finfet_canvas,
}


def generate_fine_canvas(
    architecture: str,
    rng: np.random.Generator,
    params: GenerationParams,
    preset_overrides: dict | None = None,
) -> np.ndarray:
    """Legacy single-mat canvas (one preset filling the whole fine canvas,
    no zone/strip composition). Kept for direct/simple use and tests;
    generate_sample uses generate_fine_canvas_zoned by default.
    """
    preset = get_preset(architecture)
    if preset_overrides:
        preset.update(preset_overrides)
    generator = _GENERATORS[preset["kind"]]
    return generator(
        FINE_CANVAS_SIZE_PX,
        preset,
        params.collapse_threshold_nm,
        rng,
        linewidth_bias_nm=params.linewidth_bias_nm,
        corner_rounding_px=params.corner_rounding_px,
    )


def generate_fine_canvas_zoned(
    architecture: str,
    rng: np.random.Generator,
    params: GenerationParams,
) -> dict:
    """Generate a zoned fine canvas with mat/strip composition."""
    preset = get_preset(architecture)
    return generate_zone_canvas(
        FINE_CANVAS_SIZE_PX,
        preset["kind"],
        params.collapse_threshold_nm,
        rng,
        preset_name=architecture,
        mat_size_nm=params.mat_size_nm,
        strip_width_nm=params.strip_width_nm,
        linewidth_bias_nm=params.linewidth_bias_nm,
        corner_rounding_px=params.corner_rounding_px,
    )


def _pick_crop_origin(
    zone_result: dict,
    params: GenerationParams,
    rng: np.random.Generator,
) -> tuple[int, int]:
    """Choose where to crop the reference image from the fine canvas.

    With probability `boundary_bias`, deliberately bias the crop to straddle
    a mat/strip boundary (a harder, more realistic matching scenario).
    """
    max_offset = FINE_CANVAS_SIZE_PX - REFERENCE_SIZE_PX
    strip_rects = zone_result.get("strip_rects") or []

    if strip_rects and rng.random() < params.boundary_bias:
        sx, sy, sw, sh = strip_rects[int(rng.integers(0, len(strip_rects)))]
        scx, scy = sx + sw / 2.0, sy + sh / 2.0
        x0 = scx - REFERENCE_SIZE_PX / 2.0 + rng.uniform(-250, 250)
        y0 = scy - REFERENCE_SIZE_PX / 2.0 + rng.uniform(-250, 250)
        x0 = int(np.clip(x0, 0, max_offset))
        y0 = int(np.clip(y0, 0, max_offset))
        return x0, y0

    x0 = int(rng.integers(0, max_offset + 1))
    y0 = int(rng.integers(0, max_offset + 1))
    return x0, y0


def generate_sample(
    architecture: str,
    rng: np.random.Generator,
    params: GenerationParams,
    preset_overrides: dict | None = None,
) -> dict:
    """Generate one complete Drift-Sense sample.

    Returns a dict with reference_img, search_img, ground truth coordinates,
    and all generation metadata.
    """
    if preset_overrides:
        fine_canvas = generate_fine_canvas(architecture, rng, params, preset_overrides)
        zone_result: dict = {"strip_rects": []}
    else:
        zone_result = generate_fine_canvas_zoned(architecture, rng, params)
        fine_canvas = zone_result["canvas"]

    x0, y0 = _pick_crop_origin(zone_result, params, rng)
    crop = fine_canvas[y0 : y0 + REFERENCE_SIZE_PX, x0 : x0 + REFERENCE_SIZE_PX]

    reference_img = sem_imaging.image_reference(
        crop,
        pixel_size_nm=PIXEL_SIZE_REF_NM,
        spot_size_nm=params.beam_spot_size_nm,
        dose=params.dose_reference,
        rng=rng,
        detector_noise_sigma=params.detector_noise_sigma_ref,
        drift_jitter_px=params.drift_jitter_px * 0.2,
        astigmatism_ratio=params.astigmatism_ratio,
        vignette_strength=params.vignette_strength * 0.5,
        gamma=params.gamma,
        barrel_distortion_k=params.barrel_distortion_k * 0.3,
        charging_streak_prob=params.charging_streak_prob,
        charging_streak_intensity=params.charging_streak_intensity,
        speckle_sigma=params.speckle_sigma,
        salt_pepper_prob=params.salt_pepper_prob,
        edge_brightness_gain=params.edge_brightness_gain,
        brightness_jitter=params.brightness_jitter * 0.3,
        contrast_jitter=params.contrast_jitter * 0.3,
    )

    search_img, search_transform = sem_imaging.image_search(
        fine_canvas,
        pixel_size_ref_nm=PIXEL_SIZE_REF_NM,
        pixel_size_search_nm=PIXEL_SIZE_SEARCH_NM,
        spot_size_nm=params.beam_spot_size_nm,
        dose=params.dose_search,
        rng=rng,
        shear_amplitude_px=params.shear_amplitude_px,
        drift_jitter_px=params.drift_jitter_px,
        detector_noise_sigma=params.detector_noise_sigma_search,
        astigmatism_ratio=params.astigmatism_ratio,
        vignette_strength=params.vignette_strength,
        gamma=params.gamma,
        barrel_distortion_k=params.barrel_distortion_k,
        charging_streak_prob=params.charging_streak_prob,
        charging_streak_intensity=params.charging_streak_intensity,
        speckle_sigma=params.speckle_sigma,
        salt_pepper_prob=params.salt_pepper_prob,
        edge_brightness_gain=params.edge_brightness_gain,
        rotation_deg=params.rotation_deg,
        brightness_jitter=params.brightness_jitter,
        contrast_jitter=params.contrast_jitter,
        return_transform=True,
    )

    box_w = box_h = REFERENCE_SIZE_PX // SCALE_FACTOR  # 100
    gt_x0 = x0 / SCALE_FACTOR
    gt_y0 = y0 / SCALE_FACTOR
    gt_cx, gt_cy = sem_imaging.transform_search_point(
        (gt_x0 + box_w / 2.0, gt_y0 + box_h / 2.0), search_transform
    )

    return {
        "reference_img": reference_img,
        "search_img": search_img,
        "gt_x": gt_cx,
        "gt_y": gt_cy,
        "gt_box": (gt_x0, gt_y0, box_w, box_h),
        "architecture": architecture,
        "params": params.as_dict(),
        "mat_rects": zone_result.get("mat_rects", []),
        "strip_rects": zone_result.get("strip_rects", []),
        "search_transform": search_transform,
    }
