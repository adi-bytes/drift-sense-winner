"""
Strip/Boundary Anchor Matcher — Strategy 1 for DRAM disambiguation.

Physical motivation
-------------------
Our zone canvas generator creates unique non-repeating separator strips
between DRAM array blocks. These strips have a dramatically different
spatial-frequency signature from the surrounding periodic array regions.

If a reference image contains a mat/strip boundary, that boundary is the
*only* globally unique visual landmark in the search image. By finding the
strip signature in the search image first, we can back-compute the exact
reference center with sub-pixel accuracy — completely sidestepping the
ZNCC multi-peak ambiguity problem.

Algorithm
---------
1. Detect whether the reference contains a strip boundary using row/column
   variance profiling. Strips appear as low-frequency, high-contrast bands.
2. Extract 1D intensity signature along the boundary direction.
3. Downsample signature to search-image scale and cross-correlate.
4. Back-compute reference center from the strip offset.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

logger = logging.getLogger(__name__)

STRIP_DETECTION_THRESHOLD: float = 0.3
MIN_STRIP_WIDTH_FRACTION: float = 0.05
SIGNATURE_BAND_PX: int = 20
MIN_ANCHOR_CONFIDENCE: float = 0.3
# Minimum xcorr confidence (independent of strip detection)
MIN_XCORR_CONFIDENCE: float = 0.4


@dataclass
class StripAnchorResult:
    x: float
    y: float
    confidence: float
    direction: str
    strip_pos_ref: int
    strip_pos_search: int


def _compute_variance_profile(img: np.ndarray, axis: int) -> np.ndarray:
    return np.var(img.astype(np.float32), axis=axis)


def _detect_strip_position(
    profile: np.ndarray,
    min_width_frac: float = MIN_STRIP_WIDTH_FRACTION,
) -> tuple[int | None, float]:
    if profile.size < 10:
        return None, 0.0

    min_width = max(2, int(len(profile) * min_width_frac))
    p_norm = (profile - profile.min()) / (profile.max() - profile.min() + 1e-8)
    median_var = float(np.median(p_norm))
    low_var_mask = p_norm < (median_var * (1 - STRIP_DETECTION_THRESHOLD))

    best_pos = None
    best_conf = 0.0

    in_region = False
    start = 0
    regions = []
    for i, v in enumerate(low_var_mask):
        if v and not in_region:
            start = i
            in_region = True
        elif not v and in_region:
            regions.append((start, i - 1))
            in_region = False
    if in_region:
        regions.append((start, len(low_var_mask) - 1))

    for lo, hi in regions:
        width = hi - lo + 1
        if width < min_width:
            continue
        center = (lo + hi) // 2
        strip_val = float(np.mean(p_norm[lo:hi+1]))
        left_mean = float(np.mean(p_norm[max(0, lo - width * 2):lo])) if lo > 0 else 0.0
        right_mean = float(np.mean(p_norm[hi+1:min(len(p_norm), hi + width * 2 + 1)])) if hi < len(p_norm) - 1 else 0.0
        array_mean = max(left_mean, right_mean)
        if array_mean < 1e-4:
            continue
        contrast = (array_mean - strip_val) / (array_mean + 1e-8)
        if contrast > best_conf and contrast > STRIP_DETECTION_THRESHOLD:
            best_conf = contrast
            best_pos = center

    return best_pos, best_conf


def _extract_signature(img: np.ndarray, position: int, direction: str, band_px: int = SIGNATURE_BAND_PX) -> np.ndarray:
    if direction == "horizontal":
        lo = max(0, position - band_px // 2)
        hi = min(img.shape[0], position + band_px // 2)
        return np.mean(img[lo:hi, :].astype(np.float32), axis=0)
    else:
        lo = max(0, position - band_px // 2)
        hi = min(img.shape[1], position + band_px // 2)
        return np.mean(img[:, lo:hi].astype(np.float32), axis=1)


def _find_signature_in_search(sig_ref: np.ndarray, search_profile: np.ndarray) -> tuple[int, float]:
    sig = sig_ref.astype(np.float32)
    prof = search_profile.astype(np.float32)
    sig = sig - sig.mean()
    sig_std = sig.std()
    if sig_std < 1e-6:
        return len(prof) // 2, 0.0
    sig = sig / sig_std
    corr = np.correlate(prof, sig, mode="full")
    offset = len(sig) - 1
    valid_corr = corr[offset: offset + len(prof) - len(sig) + 1]
    if valid_corr.size == 0:
        return len(prof) // 2, 0.0
    norm_factor = float(len(sig)) * float(np.std(prof)) * float(sig_std) + 1e-8
    valid_corr = valid_corr / norm_factor
    best_idx = int(np.argmax(valid_corr))
    best_pos = best_idx + len(sig) // 2
    confidence = float(valid_corr[best_idx])
    return best_pos, max(confidence, 0.0)


def strip_anchor_match(reference: np.ndarray, search: np.ndarray, zncc_candidate=None) -> StripAnchorResult | None:
    """
    Identifies deterministic boundary structures (strips) in periodic logic/DRAM arrays.
    Resolves aliasing by replacing the ambiguous ZNCC coordinate with the exact strip coordinate.
    """
    scale_factor = 10.0
    
    pred_x = zncc_candidate.x if zncc_candidate else search.shape[1] / 2.0
    pred_y = zncc_candidate.y if zncc_candidate else search.shape[0] / 2.0
    
    total_conf = 1.0
    detected = False

    for direction in ["horizontal", "vertical"]:
        axis = 1 if direction == "horizontal" else 0
        ref_var_profile = _compute_variance_profile(reference, axis)
        
        strip_pos_ref, strip_conf_ref = _detect_strip_position(ref_var_profile)
        if strip_pos_ref is None:
            continue
            
        logger.info("Strip boundary detected: %s at ref_px=%d (conf=%.2f)", direction, strip_pos_ref, strip_conf_ref)

        search_var_profile = _compute_variance_profile(search, axis)
        strip_pos_search, strip_conf_search = _detect_strip_position(search_var_profile)
        
        if strip_pos_search is None:
            continue

        if direction == "horizontal":
            # Horizontal strip means the strip spans X, so its variance position is Y.
            strip_offset_ref_y = strip_pos_ref - (reference.shape[0] / 2.0)
            pred_y = strip_pos_search - (strip_offset_ref_y / scale_factor)
        else:
            # Vertical strip means the strip spans Y, so its variance position is X.
            strip_offset_ref_x = strip_pos_ref - (reference.shape[1] / 2.0)
            pred_x = strip_pos_search - (strip_offset_ref_x / scale_factor)

        conf = float(strip_conf_ref) * float(strip_conf_search)
        total_conf *= conf
        detected = True

    if not detected or total_conf < STRIP_DETECTION_THRESHOLD * STRIP_DETECTION_THRESHOLD:
        return None

    logger.info("Merged Strip anchor: pred=(%.1f, %.1f), conf=%.3f", pred_x, pred_y, total_conf)
    
    return StripAnchorResult(
        x=float(pred_x),
        y=float(pred_y),
        confidence=total_conf,
        direction="merged",
        strip_pos_ref=0,
        strip_pos_search=0,
    )
