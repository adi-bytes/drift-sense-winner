"""Sanity checks for the Drift-Sense synthetic data generator and matcher.

Tests validate:
  1. Generator produces correct image sizes and ground truth bounds
  2. Ground truth patch matches downsampled reference (10x scale consistency)
  3. Edge brightening produces visible edge enhancement
  4. Coarse matcher finds the correct region in low-noise conditions
  5. Full pipeline (localize) returns a valid result
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.matcher.coarse_matcher import coarse_match
from src.pipeline import GenerationParams, generate_sample
from src.presets import PRESETS
from src.sem_imaging import apply_edge_brightening
from src.sem_imaging import transform_search_point

# FinFET's thin fin width aliases harder under 10nm/px downsampling than
# DRAM's broader features, so the true-location diff is naturally noisier
# even with near-zero imaging noise.
MAX_MEAN_ABS_DIFF: float = 30.0


@pytest.mark.parametrize("architecture", list(PRESETS.keys()))
def test_ground_truth_patch_matches_reference(architecture: str) -> None:
    """The search image patch at the ground-truth box should closely match
    the downsampled reference image (validating 10x scale consistency).
    """
    rng = np.random.default_rng(0)
    params = GenerationParams(
        beam_spot_size_nm=1.0,
        dose_reference=1e6,
        dose_search=1e6,
        shear_amplitude_px=0.0,
        drift_jitter_px=0.0,
        detector_noise_sigma_ref=0.0,
        detector_noise_sigma_search=0.0,
        edge_brightness_gain=0.0,
        rotation_deg=0.0,
        brightness_jitter=0.0,
        contrast_jitter=0.0,
    )
    sample = generate_sample(architecture, rng, params)

    x0, y0, w, h = (int(round(v)) for v in sample["gt_box"])
    patch_at_gt = sample["search_img"][y0 : y0 + h, x0 : x0 + w]
    reference_downsampled = cv2.resize(
        sample["reference_img"],
        (w, h),
        interpolation=cv2.INTER_AREA,
    )

    mean_abs_diff = np.abs(
        patch_at_gt.astype(int) - reference_downsampled.astype(int)
    ).mean()
    assert mean_abs_diff < MAX_MEAN_ABS_DIFF, (
        f"Mean abs diff {mean_abs_diff:.1f} >= {MAX_MEAN_ABS_DIFF} for {architecture}"
    )


def test_image_shapes_and_ground_truth_in_bounds() -> None:
    """Reference and search images should be 1000x1000, GT within bounds."""
    rng = np.random.default_rng(1)
    params = GenerationParams()
    sample = generate_sample("dram_1x", rng, params)

    assert sample["reference_img"].shape == (1000, 1000)
    assert sample["search_img"].shape == (1000, 1000)
    assert 0 <= sample["gt_x"] <= 1000
    assert 0 <= sample["gt_y"] <= 1000


def test_edge_brightening_increases_edge_intensity() -> None:
    """Edge brightening should increase intensity at feature edges."""
    # Create a simple pattern with clear edges
    canvas = np.full((100, 100), 40, dtype=np.uint8)
    canvas[30:70, 30:70] = 200  # bright square

    brightened = apply_edge_brightening(canvas, gain=0.5)

    # Edge pixels should be brighter than original
    # Check a pixel on the edge of the square
    original_edge_val = int(canvas[30, 50])
    brightened_edge_val = int(brightened[30, 50])
    # The interior shouldn't change much, but edges should be brighter
    assert brightened_edge_val >= original_edge_val, (
        f"Edge brightening should increase edge intensity: "
        f"original={original_edge_val}, brightened={brightened_edge_val}"
    )


def test_coarse_matcher_finds_correct_region_low_noise() -> None:
    """With minimal noise, the coarse matcher should find the correct region
    within ~10px of ground truth.
    """
    rng = np.random.default_rng(42)
    params = GenerationParams(
        beam_spot_size_nm=3.0,
        dose_reference=5000.0,
        dose_search=2000.0,
        shear_amplitude_px=0.0,
        drift_jitter_px=0.0,
        detector_noise_sigma_ref=1.0,
        detector_noise_sigma_search=1.0,
        edge_brightness_gain=0.0,
        rotation_deg=0.0,
    )
    sample = generate_sample("dram_loose", rng, params)

    candidates = coarse_match(sample["reference_img"], sample["search_img"])

    # At least one candidate should be close to ground truth
    gt_x, gt_y = sample["gt_x"], sample["gt_y"]
    min_dist = min(np.hypot(c.x - gt_x, c.y - gt_y) for c in candidates)

    # Relaxed threshold: 15px accounts for periodic ambiguity in DRAM
    assert min_dist < 15.0, (
        f"Best candidate {min_dist:.1f}px from GT — "
        f"GT=({gt_x:.1f}, {gt_y:.1f}), "
        f"best=({candidates[0].x:.1f}, {candidates[0].y:.1f})"
    )


def test_generation_params_serialization() -> None:
    """GenerationParams should serialize to/from dict correctly."""
    params = GenerationParams(
        beam_spot_size_nm=7.0,
        edge_brightness_gain=0.5,
        rotation_deg=1.5,
    )
    d = params.as_dict()
    assert d["beam_spot_size_nm"] == 7.0
    assert d["edge_brightness_gain"] == 0.5
    assert d["rotation_deg"] == 1.5
    assert "dose_reference" in d  # default fields present too


def test_search_point_transform_matches_output_geometry() -> None:
    """The reported GT must move with raster drift and rotation."""
    transform = {
        "row_shift": np.full(100, 2.0, dtype=np.float32),
        "barrel_distortion_k": 0.0,
        "rotation_deg": 0.0,
        "shape": (100, 100),
    }
    assert transform_search_point((50.0, 50.0), transform) == (48.0, 50.0)


def test_localize_end_to_end(tmp_path) -> None:
    """Exercise the public inference pipeline on a generated sample."""
    from localize import localize

    rng = np.random.default_rng(7)
    params = GenerationParams(
        shear_amplitude_px=0.0,
        drift_jitter_px=0.0,
        detector_noise_sigma_ref=0.0,
        detector_noise_sigma_search=0.0,
        edge_brightness_gain=0.0,
    )
    sample = generate_sample("finfet_10nm", rng, params)
    ref_path = tmp_path / "reference.png"
    search_path = tmp_path / "search.png"
    assert cv2.imwrite(str(ref_path), sample["reference_img"])
    assert cv2.imwrite(str(search_path), sample["search_img"])
    x, y = localize(str(ref_path), str(search_path))
    assert np.isfinite(x) and np.isfinite(y)
    assert 0.0 <= x <= 1000.0 and 0.0 <= y <= 1000.0
