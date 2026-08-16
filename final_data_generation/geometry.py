"""
Geometry generators for DRAM, FinFET, and zoned macro-boundary layouts.

Upgrades over src/patterns/:
  - LER/LWR: correlated 1D edge roughness applied at the geometry level BEFORE
    SEM imaging. Both Reference and Search observe the same underlying edge.
    REF: Bunday et al. SPIE 2003 — LER/LWR characterization in CD-SEM.
  - Sidewall angle: grayscale gradient inside line bodies to simulate trapezoidal
    cross-section visible in top-down SEM as brightness gradient.
    REF: Li et al. Scanning 35 (2013) — sidewall/corner effects in CD-SEM images.
  - Material SE gain: per-layer intensity multipliers (Si/poly/W/SiO2).
    REF: Li et al. Scanning 35 (2013), Joy (1995) — material-dependent SE yield.
  - Hierarchical zones: non-uniform layout with DRAM + FinFET + sparse routing.

Preserved from src/patterns/:
  - All DRAM/FinFET geometry logic
  - 320nm macro-boundary trench concept
  - Structural collapse (bridging) model
  - Corner rounding via morphological operations
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d

from final_data_generation.presets import (
    get_preset, presets_for_kind, MATERIAL_SE_GAINS
)

# ---------------------------------------------------------------------------
# Structural Defects (verbatim from src/structural_defects.py)
# REF: Pattern collapse from capillary forces in high-aspect-ratio structures.
# ---------------------------------------------------------------------------

def maybe_collapse_gap(
    gap_nm: float,
    threshold_nm: float,
    rng: np.random.Generator,
    collapse_prob: float = 0.7,
) -> bool:
    """Decide whether a gap between two adjacent lines should bridge/merge.
    Gaps at or above threshold never collapse."""
    if gap_nm >= threshold_nm:
        return False
    return bool(rng.random() < collapse_prob)


# ---------------------------------------------------------------------------
# LER/LWR: Correlated Edge Roughness Generator
# REF: Bunday et al. SPIE 2003 — LER must be applied to latent geometry,
# NOT as image noise, so both Ref and Search observe the same physical edge.
# ---------------------------------------------------------------------------

def generate_ler_roughness(
    n_positions: int,
    sigma_nm: float,
    correlation_length_nm: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate correlated 1D LER displacement for each line-edge position.

    Uses Gaussian-filtered white noise to produce a spatially correlated
    roughness process. sigma_nm controls RMS amplitude; correlation_length_nm
    controls the spatial frequency content (longer = smoother edges).

    Returns array of shape (n_positions,) with roughness displacements in nm.
    """
    if sigma_nm <= 0 or n_positions == 0:
        return np.zeros(n_positions)
    white = rng.normal(0, sigma_nm, n_positions)
    sigma_samples = max(correlation_length_nm, 1.0)
    return gaussian_filter1d(white, sigma=sigma_samples)


# ---------------------------------------------------------------------------
# Sidewall angle: brightness gradient inside line bodies
# REF: Li et al. Scanning 35 (2013) — top-down SEM contrast of trapezoidal
# cross-section shows brighter edges, slightly darker line interior center.
# ---------------------------------------------------------------------------

def apply_sidewall_gradient(
    canvas: np.ndarray,
    line_mask: np.ndarray,
    sidewall_angle_deg: float,
    axis: int = 1,
) -> np.ndarray:
    """Apply a brightness gradient inside masked line regions to simulate
    the trapezoidal sidewall cross-section visible in top-down SEM.

    A sidewall_angle_deg of 90 = vertical walls (no gradient).
    Lower angles = more trapezoidal = stronger gradient from edge to center.
    """
    if abs(sidewall_angle_deg - 90.0) < 0.5:
        return canvas
    gradient_strength = (90.0 - sidewall_angle_deg) / 90.0 * 0.25
    out = canvas.astype(np.float32)
    if axis == 0:  # horizontal lines — gradient along columns
        dist = cv2.distanceTransform((~line_mask).astype(np.uint8), cv2.DIST_L2, 3)
    else:  # vertical lines — gradient along rows
        dist = cv2.distanceTransform((~line_mask).astype(np.uint8), cv2.DIST_L2, 3)
    dist_norm = dist / (dist.max() + 1e-6)
    # Interior pixels get darkened proportional to distance from edge
    interior = line_mask.astype(np.float32)
    darkening = interior * dist_norm * gradient_strength * 60.0
    out -= darkening
    return np.clip(out, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# DRAM Generator (enhanced)
# ---------------------------------------------------------------------------

BACKGROUND: int = 40
WORD_LINE_VAL: int = 150
BIT_LINE_VAL: int = 170
CONTACT_VAL: int = 225
POSITION_JITTER_NM: float = 1.5
WIDTH_JITTER_FRACTION: float = 0.10


def _line_positions(size_px: int, pitch_nm: float, rng: np.random.Generator) -> np.ndarray:
    positions: list[float] = []
    pos = rng.uniform(0, pitch_nm)
    while pos < size_px:
        positions.append(pos)
        pos += pitch_nm + rng.normal(0, POSITION_JITTER_NM)
    return np.array(positions)


def _line_mask_with_ler(
    size_px: int,
    positions: np.ndarray,
    width_nm: float,
    collapse_threshold_nm: float,
    rng: np.random.Generator,
    width_jitter_fraction: float = WIDTH_JITTER_FRACTION,
    linewidth_bias_nm: float = 0.0,
    ler_sigma_nm: float = 0.0,
    ler_correlation_nm: float = 20.0,
) -> np.ndarray:
    """1D boolean mask with LER applied to each line edge independently.

    LER roughness is generated at the geometry level so the same physical
    edge appears in both Reference and Search images.
    REF: Bunday et al. SPIE 2003.
    """
    mask = np.zeros(size_px, dtype=bool)
    biased_width_nm = max(width_nm + linewidth_bias_nm, 1.0)
    widths = biased_width_nm * (1.0 + rng.normal(0, width_jitter_fraction, size=len(positions)))
    widths = np.clip(widths, biased_width_nm * 0.5, biased_width_nm * 1.5)

    # Generate independent LER for left and right edges of each line
    ler_left = generate_ler_roughness(len(positions), ler_sigma_nm, ler_correlation_nm, rng)
    ler_right = generate_ler_roughness(len(positions), ler_sigma_nm, ler_correlation_nm, rng)

    for i, center in enumerate(positions):
        half_w = widths[i] / 2.0
        # Apply LER: left edge shifts inward/outward, right edge independently
        lo = int(round(center - half_w + ler_left[i]))
        hi = int(round(center + half_w + ler_right[i]))
        mask[max(lo, 0): min(hi, size_px)] = True

        if i + 1 < len(positions):
            next_center = positions[i + 1]
            next_half_w = widths[i + 1] / 2.0
            gap_nm = (next_center - next_half_w) - (center + half_w)
            if maybe_collapse_gap(gap_nm, collapse_threshold_nm, rng):
                bridge_lo = int(round(center + half_w))
                bridge_hi = int(round(next_center - next_half_w))
                mask[max(bridge_lo, 0): min(bridge_hi, size_px)] = True
    return mask


def generate_dram_canvas(
    size_px: int,
    preset: dict,
    collapse_threshold_nm: float,
    rng: np.random.Generator,
    linewidth_bias_nm: float = 0.0,
    corner_rounding_px: float = 0.0,
    ler_sigma_nm: float = 0.0,
    ler_correlation_nm: float = 20.0,
    sidewall_angle_deg: float = 90.0,
) -> np.ndarray:
    """Render the DRAM cell array with LER, sidewall, and material gains."""
    canvas = np.full((size_px, size_px), BACKGROUND, dtype=np.uint8)

    word_positions = _line_positions(size_px, preset["word_line_pitch_nm"], rng)
    bit_positions = _line_positions(size_px, preset["bit_line_pitch_nm"], rng)

    row_mask = _line_mask_with_ler(
        size_px, word_positions, preset["word_line_width_nm"],
        collapse_threshold_nm, rng, linewidth_bias_nm=linewidth_bias_nm,
        ler_sigma_nm=ler_sigma_nm, ler_correlation_nm=ler_correlation_nm,
    )
    col_mask = _line_mask_with_ler(
        size_px, bit_positions, preset["bit_line_width_nm"],
        collapse_threshold_nm, rng, linewidth_bias_nm=linewidth_bias_nm,
        ler_sigma_nm=ler_sigma_nm, ler_correlation_nm=ler_correlation_nm,
    )

    # Apply material SE gain for word lines (poly-Si)
    wl_val = int(WORD_LINE_VAL * MATERIAL_SE_GAINS["word_line"])
    bl_val = int(BIT_LINE_VAL * MATERIAL_SE_GAINS["bit_line"])
    ct_val = int(CONTACT_VAL * MATERIAL_SE_GAINS["contact"])

    canvas[row_mask, :] = np.maximum(canvas[row_mask, :], wl_val)
    canvas[:, col_mask] = np.maximum(canvas[:, col_mask], bl_val)

    base_radius = max(preset["contact_diameter_nm"] + linewidth_bias_nm, 1.0) / 2.0
    for i, wl in enumerate(word_positions):
        for j, bl in enumerate(bit_positions):
            if (i + j) % 2 == 0:
                radius = max(1, int(round(base_radius * (1.0 + rng.normal(0, WIDTH_JITTER_FRACTION)))))
                cv2.circle(canvas, (int(round(bl)), int(round(wl))), radius, ct_val, -1)

    # Sidewall gradient on word lines
    canvas = apply_sidewall_gradient(
        canvas,
        np.broadcast_to(row_mask[:, None], canvas.shape),
        sidewall_angle_deg, axis=0,
    )

    if corner_rounding_px >= 0.5:
        k = max(1, int(round(corner_rounding_px)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
        canvas = cv2.morphologyEx(canvas, cv2.MORPH_OPEN, kernel)
        canvas = cv2.morphologyEx(canvas, cv2.MORPH_CLOSE, kernel)

    return canvas


# ---------------------------------------------------------------------------
# FinFET Generator (enhanced)
# ---------------------------------------------------------------------------

FIN_VAL: int = 150
GATE_VAL: int = 170


def generate_finfet_canvas(
    size_px: int,
    preset: dict,
    collapse_threshold_nm: float,
    rng: np.random.Generator,
    linewidth_bias_nm: float = 0.0,
    corner_rounding_px: float = 0.0,
    ler_sigma_nm: float = 0.0,
    ler_correlation_nm: float = 20.0,
    sidewall_angle_deg: float = 90.0,
) -> np.ndarray:
    """Render the FinFET array with LER, sidewall, and material gains."""
    canvas = np.full((size_px, size_px), BACKGROUND, dtype=np.uint8)

    fin_positions = _line_positions(size_px, preset["fin_pitch_nm"], rng)
    gate_positions = _line_positions(size_px, preset["gate_pitch_nm"], rng)

    col_mask = _line_mask_with_ler(
        size_px, fin_positions, preset["fin_width_nm"],
        collapse_threshold_nm, rng, linewidth_bias_nm=linewidth_bias_nm,
        ler_sigma_nm=ler_sigma_nm, ler_correlation_nm=ler_correlation_nm,
    )
    row_mask = _line_mask_with_ler(
        size_px, gate_positions, preset["gate_length_nm"],
        collapse_threshold_nm, rng, linewidth_bias_nm=linewidth_bias_nm,
        ler_sigma_nm=ler_sigma_nm, ler_correlation_nm=ler_correlation_nm,
    )

    fin_val = int(FIN_VAL * MATERIAL_SE_GAINS["fin"])
    gate_val = int(GATE_VAL * MATERIAL_SE_GAINS["gate"])
    ct_val = int(CONTACT_VAL * MATERIAL_SE_GAINS["contact"])

    canvas[:, col_mask] = np.maximum(canvas[:, col_mask], fin_val)
    canvas[row_mask, :] = np.maximum(canvas[row_mask, :], gate_val)

    half = max(1, int(round(max(preset["contact_size_nm"] + linewidth_bias_nm, 1.0) / 2.0)))
    for i, fin_x in enumerate(fin_positions):
        for j in range(len(gate_positions) - 1):
            if (i + j) % 2 == 0:
                mid_y = (gate_positions[j] + gate_positions[j + 1]) / 2.0
                x, y = int(round(fin_x)), int(round(mid_y))
                p0 = (max(x - half, 0), max(y - half, 0))
                p1 = (min(x + half, size_px - 1), min(y + half, size_px - 1))
                cv2.rectangle(canvas, p0, p1, ct_val, -1)

    canvas = apply_sidewall_gradient(
        canvas,
        np.broadcast_to(col_mask[None, :], canvas.shape),
        sidewall_angle_deg, axis=1,
    )

    if corner_rounding_px >= 0.5:
        k = max(1, int(round(corner_rounding_px)))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
        canvas = cv2.morphologyEx(canvas, cv2.MORPH_OPEN, kernel)
        canvas = cv2.morphologyEx(canvas, cv2.MORPH_CLOSE, kernel)

    return canvas


# ---------------------------------------------------------------------------
# Zone / Macro-boundary Layout (enhanced with hierarchical variety)
# REF: 320nm trench concept preserved. Extended with non-uniform DRAM+FinFET
# mixed-zone layout to break infinite periodic ambiguity more aggressively.
# ---------------------------------------------------------------------------

STRIP_BASE_VAL: int = 95
STRIP_LINE_VAL: int = 128
STRIP_LINE_PITCH_NM: float = 220.0
STRIP_LINE_WIDTH_NM: float = 9.0


def _strip_routing_texture(size_px: int, rng: np.random.Generator) -> np.ndarray:
    """Peripheral strip: flat mid-gray with sparse routing lines (global interconnect)."""
    canvas = np.full((size_px, size_px), STRIP_BASE_VAL, dtype=np.uint8)
    half = STRIP_LINE_WIDTH_NM / 2.0
    for axis_positions, is_row in (
        (np.arange(rng.uniform(0, STRIP_LINE_PITCH_NM), size_px, STRIP_LINE_PITCH_NM), True),
        (np.arange(rng.uniform(0, STRIP_LINE_PITCH_NM), size_px, STRIP_LINE_PITCH_NM), False),
    ):
        for center in axis_positions:
            lo = max(int(round(center - half)), 0)
            hi = min(int(round(center + half)), size_px)
            if is_row:
                canvas[lo:hi, :] = STRIP_LINE_VAL
            else:
                canvas[:, lo:hi] = STRIP_LINE_VAL
    return canvas


def _zone_grid(size_px: int, mat_size_nm: float, strip_width_nm: float) -> list:
    spans: list = []
    pos = 0.0
    is_mat = True
    while pos < size_px:
        span_len = mat_size_nm if is_mat else strip_width_nm
        end = min(pos + span_len, size_px)
        spans.append((is_mat, int(round(pos)), int(round(end))))
        pos = end
        is_mat = not is_mat
    return spans


def generate_zone_canvas(
    size_px: int,
    kind: str,
    collapse_threshold_nm: float,
    rng: np.random.Generator,
    preset_name: str | None = None,
    mat_size_nm: float = 2600.0,
    strip_width_nm: float = 320.0,
    linewidth_bias_nm: float = 0.0,
    corner_rounding_px: float = 0.0,
    ler_sigma_nm: float = 0.0,
    ler_correlation_nm: float = 20.0,
    sidewall_angle_deg: float = 90.0,
    mixed_zones: bool = True,
) -> dict:
    """Tile mats across the canvas with optional mixed DRAM+FinFET zones.

    When mixed_zones=True, alternates between DRAM and FinFET mat types
    to break the periodicity that causes matcher aliasing.
    The 320nm strip (trench) is preserved as the primary macro-boundary.
    """
    _gen_map = {"dram": generate_dram_canvas, "finfet": generate_finfet_canvas}
    presets_d = presets_for_kind("dram")
    presets_f = presets_for_kind("finfet")
    fixed_preset = get_preset(preset_name) if preset_name is not None else None

    canvas = _strip_routing_texture(size_px, rng)
    row_spans = _zone_grid(size_px, mat_size_nm, strip_width_nm)
    col_spans = _zone_grid(size_px, mat_size_nm, strip_width_nm)

    mat_rects: list = []
    strip_rects: list = []
    mat_index = 0

    for row_is_mat, y0, y1 in row_spans:
        for col_is_mat, x0, x1 in col_spans:
            if row_is_mat and col_is_mat and y1 > y0 and x1 > x0:
                mat_h, mat_w = y1 - y0, x1 - x0

                if fixed_preset is not None:
                    preset = fixed_preset
                    generator = _gen_map[preset["kind"]]
                elif mixed_zones and mat_index % 2 == 1:
                    # Alternate: odd mat index uses FinFET
                    preset = presets_f[int(rng.integers(0, len(presets_f)))]
                    generator = generate_finfet_canvas
                else:
                    # Even mat index uses DRAM (or the requested kind)
                    if kind == "finfet":
                        preset = presets_f[int(rng.integers(0, len(presets_f)))]
                        generator = generate_finfet_canvas
                    else:
                        preset = presets_d[int(rng.integers(0, len(presets_d)))]
                        generator = generate_dram_canvas

                child_rng = np.random.default_rng(rng.integers(0, 2**31 - 1))
                mat_size = max(mat_h, mat_w)
                mat_canvas = generator(
                    mat_size, preset, collapse_threshold_nm, child_rng,
                    linewidth_bias_nm=linewidth_bias_nm,
                    corner_rounding_px=corner_rounding_px,
                    ler_sigma_nm=ler_sigma_nm,
                    ler_correlation_nm=ler_correlation_nm,
                    sidewall_angle_deg=sidewall_angle_deg,
                )
                canvas[y0:y1, x0:x1] = mat_canvas[:mat_h, :mat_w]
                mat_rects.append((x0, y0, mat_w, mat_h))
                mat_index += 1
            else:
                strip_rects.append((x0, y0, x1 - x0, y1 - y0))

    return {"canvas": canvas, "mat_rects": mat_rects, "strip_rects": strip_rects}
