"""
Sub-pixel refinement for the final match location.

After coarse matching + disambiguation select the best candidate at integer
pixel resolution, this module refines the location to sub-pixel accuracy.

Method:
  1. Extract a padded patch around the best candidate in the search image.
  2. Upsample both the template and the patch by a factor (default 4x)
     using cubic interpolation.
  3. Run ZNCC at the upsampled resolution to find the refined peak.
  4. Fit a 2D parabola to the 3x3 neighborhood around the peak for
     sub-pixel offset estimation.
  5. Convert back to original search-image coordinates.

Target: <0.5 px improvement over the integer-resolution coarse match.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from src.matcher.coarse_matcher import Candidate

logger = logging.getLogger(__name__)

# Default upsampling factor for sub-pixel refinement
DEFAULT_UPSAMPLE_FACTOR: int = 4

# Padding around the candidate for patch extraction (pixels in search coords)
REFINE_PADDING_PX: int = 15


def _parabolic_offset_1d(values: np.ndarray, peak: int) -> float:
    """Fit a local 1-D parabola and return a bounded peak offset."""
    if peak <= 0 or peak >= len(values) - 1:
        return 0.0
    left, center, right = map(float, values[peak - 1 : peak + 2])
    denominator = 2.0 * (left - 2.0 * center + right)
    if abs(denominator) < 1e-8:
        return 0.0
    return float(np.clip((left - right) / denominator, -0.5, 0.5))


def refine_subpixel(
    reference: np.ndarray,
    search: np.ndarray,
    candidate: Candidate,
    search_window: int = 20,
) -> tuple[float, float]:
    """Refine a coarse 10x match with NCC and a parabolic peak fit."""
    h, w = search.shape
    tw = candidate.template_w
    th = candidate.template_h
    coarse_x = candidate.x
    coarse_y = candidate.y
    x0 = max(0, round(coarse_x - tw / 2.0) - search_window)
    y0 = max(0, round(coarse_y - th / 2.0) - search_window)
    x1 = min(w, round(coarse_x + tw / 2.0) + search_window)
    y1 = min(h, round(coarse_y + th / 2.0) + search_window)
    region = search[y0:y1, x0:x1]
    
    template_base = cv2.resize(
        reference,
        (max(round(reference.shape[1] / candidate.scale), 1),
         max(round(reference.shape[0] / candidate.scale), 1)),
        interpolation=cv2.INTER_AREA
    )
    if candidate.rotation != 0.0:
        M = cv2.getRotationMatrix2D((template_base.shape[1] / 2.0, template_base.shape[0] / 2.0), candidate.rotation, 1.0)
        abs_cos = abs(M[0, 0])
        abs_sin = abs(M[0, 1])
        new_w = int(template_base.shape[0] * abs_sin + template_base.shape[1] * abs_cos)
        new_h = int(template_base.shape[0] * abs_cos + template_base.shape[1] * abs_sin)
        M[0, 2] += new_w / 2.0 - template_base.shape[1] / 2.0
        M[1, 2] += new_h / 2.0 - template_base.shape[0] / 2.0
        template = cv2.warpAffine(template_base, M, (new_w, new_h), borderMode=cv2.BORDER_REPLICATE)
    else:
        template = template_base

    if region.shape[0] <= template.shape[0] or region.shape[1] <= template.shape[1]:
        return float(coarse_x), float(coarse_y)
    corr = cv2.matchTemplate(region, template, cv2.TM_CCOEFF_NORMED)
    _, _, _, max_loc = cv2.minMaxLoc(corr)
    dx = _parabolic_offset_1d(corr[max_loc[1], :], max_loc[0])
    dy = _parabolic_offset_1d(corr[:, max_loc[0]], max_loc[1])
    return (
        float(np.clip(x0 + max_loc[0] + template.shape[1] / 2.0 + dx, 0, w)),
        float(np.clip(y0 + max_loc[1] + template.shape[0] / 2.0 + dy, 0, h)),
    )


def _parabola_subpixel_offset(
    corr_map: np.ndarray, peak_x: int, peak_y: int
) -> tuple[float, float]:
    """Fit a 2D parabola to the 3x3 neighborhood around the correlation peak
    and return the sub-pixel offset from the integer peak location.

    Uses the standard separable approach:
      dx = (f(x-1) - f(x+1)) / (2 * (f(x-1) - 2*f(x) + f(x+1)))
    applied independently along x and y.

    Returns:
        (dx, dy) sub-pixel offset from (peak_x, peak_y).
    """
    h, w = corr_map.shape
    # Bounds check: need 1 pixel margin on all sides
    if peak_x < 1 or peak_x >= w - 1 or peak_y < 1 or peak_y >= h - 1:
        return 0.0, 0.0

    # X direction
    fx_m1 = float(corr_map[peak_y, peak_x - 1])
    fx_0 = float(corr_map[peak_y, peak_x])
    fx_p1 = float(corr_map[peak_y, peak_x + 1])
    denom_x = 2.0 * (fx_m1 - 2.0 * fx_0 + fx_p1)
    dx = (fx_m1 - fx_p1) / denom_x if abs(denom_x) > 1e-10 else 0.0

    # Y direction
    fy_m1 = float(corr_map[peak_y - 1, peak_x])
    fy_0 = float(corr_map[peak_y, peak_x])
    fy_p1 = float(corr_map[peak_y + 1, peak_x])
    denom_y = 2.0 * (fy_m1 - 2.0 * fy_0 + fy_p1)
    dy = (fy_m1 - fy_p1) / denom_y if abs(denom_y) > 1e-10 else 0.0

    # Clamp to [-0.5, 0.5] — larger offsets indicate the parabola fit is poor
    dx = max(-0.5, min(0.5, dx))
    dy = max(-0.5, min(0.5, dy))

    return dx, dy


def refine_location(
    reference: np.ndarray,
    search: np.ndarray,
    candidate: Candidate,
    upsample_factor: int = DEFAULT_UPSAMPLE_FACTOR,
    padding: int = REFINE_PADDING_PX,
) -> tuple[float, float, float]:
    """Refine a candidate match location to sub-pixel accuracy.

    Args:
        reference: Original reference image (1000x1000, grayscale uint8).
        search: Search image (1000x1000, grayscale uint8).
        candidate: Best candidate from disambiguation.
        upsample_factor: Upsampling factor for sub-pixel resolution.
        padding: Extra pixels around candidate for patch extraction.

    Returns:
        (refined_x, refined_y, refined_score) in original search-image coords.
    """
    h, w = search.shape

    # Extract search patch around candidate (with padding)
    half_w = candidate.template_w / 2.0 + padding
    half_h = candidate.template_h / 2.0 + padding

    x0 = max(round(candidate.x - half_w), 0)
    y0 = max(round(candidate.y - half_h), 0)
    x1 = min(round(candidate.x + half_w), w)
    y1 = min(round(candidate.y + half_h), h)

    patch = search[y0:y1, x0:x1]

    if patch.size == 0:
        logger.warning("Empty patch for refinement — returning coarse location")
        return candidate.x, candidate.y, candidate.score

    # Resize reference to template size
    template = cv2.resize(
        reference,
        (candidate.template_w, candidate.template_h),
        interpolation=cv2.INTER_AREA,
    )

    # Check that template fits in patch
    if template.shape[0] >= patch.shape[0] or template.shape[1] >= patch.shape[1]:
        logger.warning("Template larger than patch — returning coarse location")
        return candidate.x, candidate.y, candidate.score

    # Upsample both
    up_template = cv2.resize(
        template,
        (template.shape[1] * upsample_factor, template.shape[0] * upsample_factor),
        interpolation=cv2.INTER_CUBIC,
    )
    up_patch = cv2.resize(
        patch,
        (patch.shape[1] * upsample_factor, patch.shape[0] * upsample_factor),
        interpolation=cv2.INTER_CUBIC,
    )

    # Check dimensions after upsampling
    if (
        up_template.shape[0] >= up_patch.shape[0]
        or up_template.shape[1] >= up_patch.shape[1]
    ):
        logger.warning("Upsampled template too large — returning coarse location")
        return candidate.x, candidate.y, candidate.score

    # Run ZNCC at upsampled resolution
    corr_map = cv2.matchTemplate(up_patch, up_template, cv2.TM_CCOEFF_NORMED)

    _, max_score, _, max_loc = cv2.minMaxLoc(corr_map)
    peak_x, peak_y = max_loc

    # Sub-pixel offset from parabola fit
    dx, dy = _parabola_subpixel_offset(corr_map, peak_x, peak_y)

    # Convert back to original search-image coordinates
    # The peak is at (peak_x + dx) in the upsampled patch coordinate system.
    # Template center offset in upsampled coords:
    up_tw = up_template.shape[1]
    up_th = up_template.shape[0]

    refined_x_in_patch = (peak_x + dx + up_tw / 2.0) / upsample_factor
    refined_y_in_patch = (peak_y + dy + up_th / 2.0) / upsample_factor

    refined_x = x0 + refined_x_in_patch
    refined_y = y0 + refined_y_in_patch

    # Clamp to image bounds
    refined_x = max(0.0, min(float(w), refined_x))
    refined_y = max(0.0, min(float(h), refined_y))

    logger.info(
        "Refinement: (%.2f, %.2f) -> (%.2f, %.2f), "
        "delta=(%.3f, %.3f), score=%.4f -> %.4f",
        candidate.x,
        candidate.y,
        refined_x,
        refined_y,
        refined_x - candidate.x,
        refined_y - candidate.y,
        candidate.score,
        max_score,
    )

    return refined_x, refined_y, float(max_score)
