"""
CLI entry point for the upgraded synthetic SEM dataset generator.

Usage:
    # Quick start with severity level (recommended):
    python final_data_generation/run.py --num-samples 50 --severity-level 3

    # Manual parameter control:
    python final_data_generation/run.py --num-samples 30 --dose-search 400 \
        --drift-amplitude-px 2.0 --ler-sigma-nm 2.5 --boundary-bias 1.0

    # Full extreme-difficulty dataset:
    python final_data_generation/run.py --num-samples 50 --severity-level 6 \
        --output-dir ./data_final --seed 42
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass

import cv2
import numpy as np

# Self-contained imports from within the package
from final_data_generation.presets import (
    PRESETS, get_preset, presets_for_kind, get_severity_params
)
from final_data_generation.geometry import generate_zone_canvas
from final_data_generation.sem_physics import (
    image_reference, image_search, transform_search_point, SearchTransform
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

REFERENCE_SIZE_PX: int = 1000
PIXEL_SIZE_REF_NM: int = 1
PIXEL_SIZE_SEARCH_NM: int = 10
SCALE_FACTOR: int = PIXEL_SIZE_SEARCH_NM // PIXEL_SIZE_REF_NM   # 10

# Fine canvas: large enough that a 10000×10000 search region fits even when
# the reference center is jittered ±300 fine pixels from canvas center.
# Min canvas = 10000 (search) + 2×300 (jitter margin) = 10600 → use 12000.
# REF: hackathon spec slide 18 — "generate a larger continuous version of the
# die layout, then resize that larger region down to 1000×1000".
FINE_CANVAS_SIZE_PX: int = 12000
SEARCH_FINE_SIZE_PX: int = REFERENCE_SIZE_PX * SCALE_FACTOR  # 10000 fine px = 10 um FOV


# ---------------------------------------------------------------------------
# Generation Parameters
# ---------------------------------------------------------------------------

@dataclass
class GenerationParams:
    """All tuneable parameters for the upgraded synthetic SEM pipeline.

    Split into:
      - Process/geometry parameters (describe the actual chip structure)
      - Instrument/acquisition parameters (describe how the SEM observes it)
    This separation follows the conceptual framework of Villarrubia et al.
    SPIE 5038 (2003): G = ProcessGeometry(theta_process),
    I = SEM_Acquisition(G, theta_instrument).
    """

    # === PROCESS / GEOMETRY PARAMETERS ===
    collapse_threshold_nm: float = 10.0
    linewidth_bias_nm: float = 0.0
    corner_rounding_px: float = 0.0
    mat_size_nm: float = 2600.0
    strip_width_nm: float = 320.0
    boundary_bias: float = 0.35
    mixed_zones: bool = True

    # LER/LWR — applied at geometry level BEFORE imaging
    # REF: Bunday et al. SPIE 2003
    ler_sigma_nm: float = 0.0
    ler_correlation_nm: float = 20.0

    # Sidewall angle (90=vertical walls, lower=trapezoidal)
    # REF: Li et al. Scanning 2013
    sidewall_angle_deg: float = 90.0

    # === INSTRUMENT / ACQUISITION PARAMETERS ===
    beam_spot_size_nm: float = 5.0
    astigmatism_ratio: float = 1.0

    # Dose: proxy for beam_current * dwell_time
    # REF: Villarrubia et al. SPIE 5038 (2003)
    dose_reference: float = 2000.0
    dose_search: float = 200.0

    # Smooth temporal drift trajectory (new: replaces linear shear)
    # REF: Maraghechi et al. Ultramicroscopy 187 (2018)
    drift_amplitude_px: float = 1.5
    drift_correlation_rows: float = 50.0

    # Correlated scan-line shifts (separate from drift)
    # REF: Maraghechi et al. Mechanics of Materials
    scanline_shift_sigma_px: float = 0.5
    scanline_shift_correlation: float = 5.0

    # Correlated electronic noise (replaces white Gaussian)
    # REF: Villarrubia et al. SPIE 5038 (2003)
    correlated_noise_sigma: float = 5.0
    correlated_noise_length_px: float = 2.0

    # Detector response (replaces gamma as primary nonlinearity)
    # REF: Li et al. Scanning 2013
    detector_gain: float = 1.0
    detector_offset: float = 0.0
    detector_nonlinearity: float = 1.0

    # Minor robustness augmentations (deprioritized per research review)
    speckle_sigma: float = 0.02
    salt_pepper_prob: float = 0.001

    # Other acquisition effects
    vignette_strength: float = 0.0
    barrel_distortion_k: float = 0.0
    charging_streak_prob: float = 0.0
    charging_streak_intensity: float = 0.0
    edge_brightness_gain: float = 0.3
    rotation_deg: float = 0.0
    brightness_jitter: float = 0.0
    contrast_jitter: float = 0.0

    def as_dict(self) -> dict:
        return asdict(self)


def _apply_severity(params: GenerationParams, level: int) -> GenerationParams:
    """Override params with a severity curriculum bundle."""
    sv = get_severity_params(level)
    params.dose_reference = sv["dose_reference"]
    params.dose_search = sv["dose_search"]
    params.drift_amplitude_px = sv["drift_amplitude_px"]
    params.drift_correlation_rows = sv["drift_correlation_rows"]
    params.scanline_shift_sigma_px = sv["scanline_shift_sigma_px"]
    params.scanline_shift_correlation = sv["scanline_shift_correlation"]
    params.correlated_noise_sigma = sv["correlated_noise_sigma"]
    params.correlated_noise_length_px = sv["correlated_noise_length_px"]
    params.ler_sigma_nm = sv["ler_sigma_nm"]
    params.ler_correlation_nm = sv["ler_correlation_nm"]
    params.sidewall_angle_deg = sv["sidewall_angle_deg"]
    params.beam_spot_size_nm = sv["beam_spot_size_nm"]
    params.astigmatism_ratio = sv["astigmatism_ratio"]
    params.vignette_strength = sv["vignette_strength"]
    params.barrel_distortion_k = sv["barrel_distortion_k"]
    params.charging_streak_prob = sv["charging_streak_prob"]
    params.speckle_sigma = sv["speckle_sigma"]
    params.salt_pepper_prob = sv["salt_pepper_prob"]
    params.boundary_bias = sv["boundary_bias"]
    return params


# Maximum jitter of the reference center from the canvas center, in fine pixels.
# ±300 fine px = ±30 search px = ±3 um — realistic stage placement uncertainty.
# Keeping it small ensures the reference always appears near (500, 500) in the
# search image after downsampling, satisfying the spec rule:
# "if more than one region matches, whichever is closest to the search image center."
_REF_JITTER_FINE_PX: int = 300


def _pick_crop_origin(rng: np.random.Generator) -> tuple[int, int]:
    """Place the reference crop near the canvas center with a small random jitter.

    The canvas center is at (FINE_CANVAS_SIZE_PX // 2, FINE_CANVAS_SIZE_PX // 2).
    The reference crop (1000×1000) is positioned so its center ≈ canvas center.
    The search region (10000×10000) is then centered on the same point, so the
    reference center maps to search-image pixel (500, 500) before drift.

    REF: Hackathon spec — reference pattern should appear near the center of the
    wide search image; disambiguation uses center-proximity as the tiebreaker.
    """
    canvas_center = FINE_CANVAS_SIZE_PX // 2
    jitter_x = int(rng.integers(-_REF_JITTER_FINE_PX, _REF_JITTER_FINE_PX + 1))
    jitter_y = int(rng.integers(-_REF_JITTER_FINE_PX, _REF_JITTER_FINE_PX + 1))
    x0 = canvas_center - REFERENCE_SIZE_PX // 2 + jitter_x
    y0 = canvas_center - REFERENCE_SIZE_PX // 2 + jitter_y
    return x0, y0


def generate_sample(
    architecture: str,
    rng: np.random.Generator,
    params: GenerationParams,
    rotation_deg: float = 0.0,
) -> dict:
    """Generate one complete upgraded Drift-Sense sample.

    Physics flow (corrected to match hackathon spec):
    1.  Generate latent wafer geometry on a 12000×12000 fine canvas
        (LER, sidewall gradient, material SE gains all applied at geometry level)
    2.  Reference crop (1000×1000 @ 1 nm/px) taken from NEAR CANVAS CENTER
        → maps to search image pixel ≈ (500, 500) before drift
    3.  Search fine patch: 10000×10000 from same canvas, centered on same point
        → apply SEM imaging independently, then downsample 10× to 1000×1000
    4.  Ground truth = reference center position in the drifted search image
        ≈ (500 ± drift, 500 ± drift) — always near center by construction

    REF: Hackathon spec slide 18 — "generate a larger continuous version of the
    same die layout, then resize that larger region down to 1000×1000."
    REF: Hackathon spec — "whichever is closest to the search image's center" —
    this is the localization TARGET DEFINITION, not just a tie-break hint.
    REF: Villarrubia et al. SPIE 5038 (2003) — Ref and Search share the same
    latent geometry but use independent instrument parameters.
    """
    preset = get_preset(architecture)

    # 1. Generate full 12000×12000 latent geometry
    zone_result = generate_zone_canvas(
        FINE_CANVAS_SIZE_PX,
        preset["kind"],
        params.collapse_threshold_nm,
        rng,
        preset_name=architecture,
        mat_size_nm=params.mat_size_nm,
        strip_width_nm=params.strip_width_nm,
        linewidth_bias_nm=params.linewidth_bias_nm,
        corner_rounding_px=params.corner_rounding_px,
        ler_sigma_nm=params.ler_sigma_nm,
        ler_correlation_nm=params.ler_correlation_nm,
        sidewall_angle_deg=params.sidewall_angle_deg,
        mixed_zones=params.mixed_zones,
    )
    fine_canvas = zone_result["canvas"]

    # 2. Reference crop — always near canvas center
    x0_ref, y0_ref = _pick_crop_origin(rng)
    crop = fine_canvas[y0_ref: y0_ref + REFERENCE_SIZE_PX,
                       x0_ref: x0_ref + REFERENCE_SIZE_PX]

    # 3a. Search fine patch: 10000×10000 centered on the same reference center
    ref_cx_fine = x0_ref + REFERENCE_SIZE_PX // 2
    ref_cy_fine = y0_ref + REFERENCE_SIZE_PX // 2
    half_search = SEARCH_FINE_SIZE_PX // 2  # 5000
    sx0 = ref_cx_fine - half_search  # left edge of search fine patch
    sy0 = ref_cy_fine - half_search  # top  edge of search fine patch

    # Clip to canvas (shouldn't clip with 12000 canvas and ±300 jitter, but be safe)
    sx0 = int(np.clip(sx0, 0, FINE_CANVAS_SIZE_PX - SEARCH_FINE_SIZE_PX))
    sy0 = int(np.clip(sy0, 0, FINE_CANVAS_SIZE_PX - SEARCH_FINE_SIZE_PX))
    search_fine_patch = fine_canvas[sy0: sy0 + SEARCH_FINE_SIZE_PX,
                                    sx0: sx0 + SEARCH_FINE_SIZE_PX]

    # 3b. Reference SEM acquisition (high dose, minimal drift)
    reference_img = image_reference(
        crop,
        pixel_size_nm=PIXEL_SIZE_REF_NM,
        spot_size_nm=params.beam_spot_size_nm,
        dose=params.dose_reference,
        rng=rng,
        correlated_noise_sigma=params.correlated_noise_sigma * 0.4,
        correlated_noise_length_px=params.correlated_noise_length_px,
        drift_amplitude_px=params.drift_amplitude_px,
        drift_correlation_rows=params.drift_correlation_rows,
        scanline_shift_sigma_px=params.scanline_shift_sigma_px,
        scanline_shift_correlation=params.scanline_shift_correlation,
        astigmatism_ratio=params.astigmatism_ratio,
        vignette_strength=params.vignette_strength * 0.5,
        barrel_distortion_k=params.barrel_distortion_k * 0.3,
        charging_streak_prob=params.charging_streak_prob,
        charging_streak_intensity=params.charging_streak_intensity,
        speckle_sigma=params.speckle_sigma,
        salt_pepper_prob=params.salt_pepper_prob,
        edge_brightness_gain=params.edge_brightness_gain,
        detector_gain=params.detector_gain,
        detector_offset=params.detector_offset,
        detector_nonlinearity=params.detector_nonlinearity,
        brightness_jitter=params.brightness_jitter * 0.3,
        contrast_jitter=params.contrast_jitter * 0.3,
    )

    # 3c. Search SEM acquisition (low dose, heavy drift) on the 10000×10000 patch
    # image_search will PSF-blur then area-average downsample 10× → 1000×1000
    search_img, search_transform = image_search(
        search_fine_patch,
        pixel_size_ref_nm=PIXEL_SIZE_REF_NM,
        pixel_size_search_nm=PIXEL_SIZE_SEARCH_NM,
        spot_size_nm=params.beam_spot_size_nm,
        dose=params.dose_search,
        rng=rng,
        correlated_noise_sigma=params.correlated_noise_sigma,
        correlated_noise_length_px=params.correlated_noise_length_px,
        drift_amplitude_px=params.drift_amplitude_px,
        drift_correlation_rows=params.drift_correlation_rows,
        scanline_shift_sigma_px=params.scanline_shift_sigma_px,
        scanline_shift_correlation=params.scanline_shift_correlation,
        astigmatism_ratio=params.astigmatism_ratio,
        vignette_strength=params.vignette_strength,
        barrel_distortion_k=params.barrel_distortion_k,
        charging_streak_prob=params.charging_streak_prob,
        charging_streak_intensity=params.charging_streak_intensity,
        speckle_sigma=params.speckle_sigma,
        salt_pepper_prob=params.salt_pepper_prob,
        edge_brightness_gain=params.edge_brightness_gain,
        rotation_deg=rotation_deg,
        detector_gain=params.detector_gain,
        detector_offset=params.detector_offset,
        detector_nonlinearity=params.detector_nonlinearity,
        brightness_jitter=params.brightness_jitter,
        contrast_jitter=params.contrast_jitter,
        return_transform=True,
    )

    # 4. Ground truth: reference center in search image coords
    # Before drift: ref center is at (5000, 5000) in the 10000×10000 fine patch,
    # which maps to search px (500, 500). Drift shifts this slightly.
    box_w = box_h = REFERENCE_SIZE_PX // SCALE_FACTOR  # 100 px in search image

    # Reference center in the search fine patch (before drift)
    ref_cx_in_patch = ref_cx_fine - sx0   # = 5000 (or close, after clipping)
    ref_cy_in_patch = ref_cy_fine - sy0
    # Convert to search image pixel coords (undrifted)
    gt_x0_search = ref_cx_in_patch / SCALE_FACTOR - box_w / 2.0
    gt_y0_search = ref_cy_in_patch / SCALE_FACTOR - box_h / 2.0
    # Apply drift transform to get the actual (drifted) center position
    gt_cx, gt_cy = transform_search_point(
        (gt_x0_search + box_w / 2.0, gt_y0_search + box_h / 2.0),
        search_transform,
    )

    return {
        "reference_img": reference_img,
        "search_img": search_img,
        "gt_x": gt_cx,
        "gt_y": gt_cy,
        "gt_box": (gt_x0_search, gt_y0_search, box_w, box_h),
        "architecture": architecture,
        "params": params.as_dict(),
        "mat_rects": zone_result.get("mat_rects", []),
        "strip_rects": zone_result.get("strip_rects", []),
        "search_transform": search_transform,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--num-samples", type=int, default=30)
    p.add_argument("--architectures", nargs="+", default=list(PRESETS.keys()), choices=list(PRESETS.keys()))
    p.add_argument("--split", default="test")
    p.add_argument("--output-dir", default="./data_final")
    p.add_argument("--seed", type=int, default=42)

    # Severity curriculum (the easy knob)
    p.add_argument(
        "--severity-level", type=int, default=None, choices=range(7),
        help="Physics severity 0=clean, 6=extreme. Overrides individual physics params.",
    )

    # Individual physics overrides (used when severity-level is not set)
    p.add_argument("--dose-reference", type=float, default=GenerationParams.dose_reference)
    p.add_argument("--dose-search", type=float, default=GenerationParams.dose_search)
    p.add_argument("--beam-spot-size-nm", type=float, default=GenerationParams.beam_spot_size_nm)
    p.add_argument("--drift-amplitude-px", type=float, default=GenerationParams.drift_amplitude_px)
    p.add_argument("--drift-correlation-rows", type=float, default=GenerationParams.drift_correlation_rows)
    p.add_argument("--scanline-shift-sigma-px", type=float, default=GenerationParams.scanline_shift_sigma_px)
    p.add_argument("--scanline-shift-correlation", type=float, default=GenerationParams.scanline_shift_correlation)
    p.add_argument("--correlated-noise-sigma", type=float, default=GenerationParams.correlated_noise_sigma)
    p.add_argument("--correlated-noise-length-px", type=float, default=GenerationParams.correlated_noise_length_px)
    p.add_argument("--ler-sigma-nm", type=float, default=GenerationParams.ler_sigma_nm)
    p.add_argument("--ler-correlation-nm", type=float, default=GenerationParams.ler_correlation_nm)
    p.add_argument("--sidewall-angle-deg", type=float, default=GenerationParams.sidewall_angle_deg)
    p.add_argument("--astigmatism-ratio", type=float, default=GenerationParams.astigmatism_ratio)
    p.add_argument("--vignette-strength", type=float, default=GenerationParams.vignette_strength)
    p.add_argument("--barrel-distortion-k", type=float, default=GenerationParams.barrel_distortion_k)
    p.add_argument("--charging-streak-prob", type=float, default=GenerationParams.charging_streak_prob)
    p.add_argument("--charging-streak-intensity", type=float, default=GenerationParams.charging_streak_intensity)
    p.add_argument("--speckle-sigma", type=float, default=GenerationParams.speckle_sigma)
    p.add_argument("--salt-pepper-prob", type=float, default=GenerationParams.salt_pepper_prob)
    p.add_argument("--boundary-bias", type=float, default=GenerationParams.boundary_bias)
    p.add_argument("--mat-size-nm", type=float, default=GenerationParams.mat_size_nm)
    p.add_argument("--strip-width-nm", type=float, default=GenerationParams.strip_width_nm)
    p.add_argument("--rotation-max-deg", type=float, default=0.0)
    p.add_argument("--no-mixed-zones", action="store_true", help="Disable mixed DRAM+FinFET zones")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    split_dir = os.path.join(args.output_dir, args.split)
    ref_dir = os.path.join(split_dir, "reference")
    search_dir = os.path.join(split_dir, "search")
    os.makedirs(ref_dir, exist_ok=True)
    os.makedirs(search_dir, exist_ok=True)

    manifest_path = os.path.join(split_dir, "manifest.csv")
    fieldnames = [
        "id", "reference_path", "search_path", "gt_x", "gt_y",
        "gt_box_x", "gt_box_y", "gt_box_w", "gt_box_h",
        "architecture", "severity_level", "sample_seed",
        "dose_reference", "dose_search", "beam_spot_size_nm",
        "drift_amplitude_px", "drift_correlation_rows",
        "scanline_shift_sigma_px", "scanline_shift_correlation",
        "correlated_noise_sigma", "correlated_noise_length_px",
        "ler_sigma_nm", "ler_correlation_nm", "sidewall_angle_deg",
        "astigmatism_ratio", "vignette_strength", "barrel_distortion_k",
        "charging_streak_prob", "boundary_bias", "rotation_deg",
    ]

    t_start = time.perf_counter()
    severity_level = args.severity_level
    if severity_level is not None:
        logger.info("Using severity curriculum level %d", severity_level)

    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()

        for i in range(args.num_samples):
            sample_seed = int(rng.integers(0, 2_000_000_000))
            sample_rng = np.random.default_rng(sample_seed)
            architecture = args.architectures[int(rng.integers(0, len(args.architectures)))]

            rotation_deg = 0.0
            if args.rotation_max_deg > 0:
                rotation_deg = float(sample_rng.uniform(-args.rotation_max_deg, args.rotation_max_deg))

            params = GenerationParams(
                dose_reference=args.dose_reference,
                dose_search=args.dose_search,
                beam_spot_size_nm=args.beam_spot_size_nm,
                drift_amplitude_px=args.drift_amplitude_px,
                drift_correlation_rows=args.drift_correlation_rows,
                scanline_shift_sigma_px=args.scanline_shift_sigma_px,
                scanline_shift_correlation=args.scanline_shift_correlation,
                correlated_noise_sigma=args.correlated_noise_sigma,
                correlated_noise_length_px=args.correlated_noise_length_px,
                ler_sigma_nm=args.ler_sigma_nm,
                ler_correlation_nm=args.ler_correlation_nm,
                sidewall_angle_deg=args.sidewall_angle_deg,
                astigmatism_ratio=args.astigmatism_ratio,
                vignette_strength=args.vignette_strength,
                barrel_distortion_k=args.barrel_distortion_k,
                charging_streak_prob=args.charging_streak_prob,
                charging_streak_intensity=args.charging_streak_intensity,
                speckle_sigma=args.speckle_sigma,
                salt_pepper_prob=args.salt_pepper_prob,
                boundary_bias=args.boundary_bias,
                mat_size_nm=args.mat_size_nm,
                strip_width_nm=args.strip_width_nm,
                mixed_zones=not args.no_mixed_zones,
            )

            # Apply severity curriculum if requested (overrides individual params)
            if severity_level is not None:
                params = _apply_severity(params, severity_level)

            sample = generate_sample(architecture, sample_rng, params, rotation_deg)

            ref_fname = f"{i:05d}.png"
            search_fname = f"{i:05d}.png"
            cv2.imwrite(os.path.join(ref_dir, ref_fname), sample["reference_img"])
            cv2.imwrite(os.path.join(search_dir, search_fname), sample["search_img"])

            gx0, gy0, gw, gh = sample["gt_box"]
            writer.writerow({
                "id": i,
                "reference_path": os.path.join("reference", ref_fname),
                "search_path": os.path.join("search", search_fname),
                "gt_x": sample["gt_x"],
                "gt_y": sample["gt_y"],
                "gt_box_x": gx0,
                "gt_box_y": gy0,
                "gt_box_w": gw,
                "gt_box_h": gh,
                "architecture": architecture,
                "severity_level": severity_level if severity_level is not None else "custom",
                "sample_seed": sample_seed,
                "dose_reference": params.dose_reference,
                "dose_search": params.dose_search,
                "beam_spot_size_nm": params.beam_spot_size_nm,
                "drift_amplitude_px": params.drift_amplitude_px,
                "drift_correlation_rows": params.drift_correlation_rows,
                "scanline_shift_sigma_px": params.scanline_shift_sigma_px,
                "scanline_shift_correlation": params.scanline_shift_correlation,
                "correlated_noise_sigma": params.correlated_noise_sigma,
                "correlated_noise_length_px": params.correlated_noise_length_px,
                "ler_sigma_nm": params.ler_sigma_nm,
                "ler_correlation_nm": params.ler_correlation_nm,
                "sidewall_angle_deg": params.sidewall_angle_deg,
                "astigmatism_ratio": params.astigmatism_ratio,
                "vignette_strength": params.vignette_strength,
                "barrel_distortion_k": params.barrel_distortion_k,
                "charging_streak_prob": params.charging_streak_prob,
                "boundary_bias": params.boundary_bias,
                "rotation_deg": rotation_deg,
            })
            logger.info("[%d/%d] %s → gt=(%.1f, %.1f)", i + 1, args.num_samples, architecture, sample["gt_x"], sample["gt_y"])

    elapsed = time.perf_counter() - t_start
    logger.info("Wrote %d samples to %s (%.1fs total, %.2fs/sample)", args.num_samples, split_dir, elapsed, elapsed / max(args.num_samples, 1))


if __name__ == "__main__":
    main()
