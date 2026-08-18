"""
Phase-Based Period Disambiguator — Strategy 2 for DRAM disambiguation.

Physical motivation
-------------------
In a perfectly periodic DRAM array with pitch P (in search pixels), the
ZNCC correlation map has identical peaks at positions:
    x_n = x_0 + n * P   for n = 0, ±1, ±2, ...

The center-proximity tie-breaker picks peak n=0 closest to 500, which
fails when the true target has large stage drift.

The key insight: the reference image is *phase-locked* to the DRAM array.
The sub-period phase offset φ of the reference center within one pitch
period is uniquely computable from the reference image's autocorrelation.

In the search image, all ZNCC peaks share the same true phase — but only
one of them has the correct *absolute* position that matches φ modulo P.

By computing φ_ref from the reference and matching it to φ_search(candidate)
for each ZNCC peak, we can select the correct peak regardless of its
distance from center.

Algorithm
---------
1. Compute the 2D autocorrelation of the reference to find dominant pitch P.
2. Extract sub-period phase offset φ_ref of the reference center.
3. For each ZNCC candidate, compute their phase φ_candidate mod P.
4. Select the candidate whose phase best matches φ_ref (mod P).
5. Fall back to center-proximity if no clear phase match exists.
"""
from __future__ import annotations

import logging

import cv2
import numpy as np

from src.matcher.coarse_matcher import Candidate

logger = logging.getLogger(__name__)

# Minimum autocorrelation peak ratio to trust the period estimate
MIN_PERIOD_CONFIDENCE: float = 0.5

# Maximum allowable phase error (in search pixels) to call a match
MAX_PHASE_ERROR_PX: float = 3.0

# Range of periods (in reference pixels) to search for
MIN_PERIOD_REF_PX: int = 40   # = 4 search px minimum period (ignore sub-feature noise)
MAX_PERIOD_REF_PX: int = 400

# Minimum period in SEARCH pixel space — prevents reacting to sub-feature periodicity
# e.g. finfet_45nm fins have 140nm pitch = 14 search px, too small to reliably phase-match
MIN_PERIOD_SEARCH_PX: float = 18.0


def _find_dominant_period(img: np.ndarray, axis: int) -> tuple[float, float]:
    """
    Find the dominant spatial period of an image along a given axis using
    the 1D power spectral density.

    Returns (period_px, confidence) where period_px is in image pixels
    and confidence is the relative height of the dominant spectral peak.
    """
    # Use the mean profile along the specified axis
    profile = np.mean(img.astype(np.float32), axis=1 - axis)
    profile -= profile.mean()

    n = len(profile)
    # Power spectrum
    fft = np.fft.rfft(profile, n=n * 4)  # Zero-pad for finer frequency resolution
    psd = np.abs(fft) ** 2
    freqs = np.fft.rfftfreq(n * 4, d=1.0)

    # Search only in the valid period range
    min_freq = 1.0 / MAX_PERIOD_REF_PX
    max_freq = 1.0 / MIN_PERIOD_REF_PX
    valid = (freqs >= min_freq) & (freqs <= max_freq)

    if not np.any(valid):
        return 0.0, 0.0

    valid_psd = psd.copy()
    valid_psd[~valid] = 0.0

    # Find peak
    peak_idx = int(np.argmax(valid_psd))
    peak_freq = float(freqs[peak_idx])
    if peak_freq < 1e-8:
        return 0.0, 0.0

    period = 1.0 / peak_freq
    # Confidence: ratio of dominant peak to second-best peak in valid range
    peak_val = float(psd[peak_idx])
    psd_copy = valid_psd.copy()
    # Zero out peak and neighbors
    nbr = max(1, int(len(freqs) / (MAX_PERIOD_REF_PX * 2)))
    psd_copy[max(0, peak_idx - nbr):peak_idx + nbr + 1] = 0.0
    second_val = float(np.max(psd_copy)) if np.any(psd_copy > 0) else 0.0
    confidence = peak_val / (second_val + peak_val + 1e-8)

    return period, confidence


def _compute_phase_offset(profile: np.ndarray, period_px: float) -> float:
    """
    Compute the sub-period phase offset of a 1D profile relative to period_px.
    Uses the complex exponential phase at the dominant frequency.
    Returns phase in [0, period_px).
    """
    if period_px <= 0:
        return 0.0
    freq = 1.0 / period_px
    n = len(profile)
    t = np.arange(n, dtype=np.float32)
    # Project onto complex exponential at this frequency
    phasor = np.dot(profile.astype(np.float32) - profile.mean(), np.exp(-2j * np.pi * freq * t))
    phase_rad = float(np.angle(phasor))
    # Convert to pixel offset
    phase_px = (phase_rad / (2 * np.pi)) * period_px
    # Normalize to [0, period_px)
    phase_px = phase_px % period_px
    return float(phase_px)


def phase_disambiguate(
    reference: np.ndarray,
    search: np.ndarray,
    candidates: list[Candidate],
    scale_factor: int = 10,
) -> tuple[Candidate | None, float]:
    """
    Select the correct candidate from a list of ZNCC peaks using phase analysis.

    Returns (best_candidate, confidence) or (None, 0.0) if phase disambiguation
    cannot be performed with sufficient confidence.

    Args:
        reference: Reference image (1000×1000 @ 1nm/px).
        search: Search image (1000×1000 @ 10nm/px).
        candidates: ZNCC candidate list (already sorted by score descending).
        scale_factor: Ratio between search and reference pixel sizes (10).
    """
    if not candidates:
        return None, 0.0

    ref_proc = cv2.normalize(reference, None, 0, 255, cv2.NORM_MINMAX).astype(np.float32)
    cv2.normalize(search, None, 0, 255, cv2.NORM_MINMAX).astype(np.float32)

    best_candidate: Candidate | None = None
    best_conf = 0.0

    for axis, axis_name in [(0, "x"), (1, "y")]:
        # Find dominant period in reference (in reference pixels)
        period_ref_px, period_conf = _find_dominant_period(ref_proc, axis=axis)
        if period_conf < MIN_PERIOD_CONFIDENCE or period_ref_px < MIN_PERIOD_REF_PX:
            logger.debug("No clear %s-axis periodicity in reference (conf=%.2f, period=%.1f)", axis_name, period_conf, period_ref_px)
            continue

        # Convert period to search pixel space
        period_search_px = period_ref_px / scale_factor

        # Skip periods too fine to reliably disambiguate in search coordinates
        if period_search_px < MIN_PERIOD_SEARCH_PX:
            logger.debug("Period %.2f search_px too fine (min %.1f), skipping %s axis",
                         period_search_px, MIN_PERIOD_SEARCH_PX, axis_name)
            continue

        logger.info("Detected %s-axis period: %.1f ref_px (%.2f search_px), conf=%.2f",
                    axis_name, period_ref_px, period_search_px, period_conf)

        # Compute phase of reference center
        if axis == 0:  # x-axis: use column-mean profile
            ref_profile = np.mean(ref_proc, axis=0)
        else:  # y-axis: use row-mean profile
            ref_profile = np.mean(ref_proc, axis=1)

        phi_ref = _compute_phase_offset(ref_profile, period_ref_px)
        logger.debug("Reference phase offset: %.2f ref_px (%.2f search_px)", phi_ref, phi_ref / scale_factor)
        phi_ref_search = phi_ref / scale_factor  # in search pixel units

        # For each candidate, compute its phase in the search image
        # Candidate position x (or y) modulo period should match phi_ref_search
        for cand in candidates:
            cand_coord = cand.x if axis == 0 else cand.y
            cand_phase = cand_coord % period_search_px
            # Phase error (account for wraparound)
            phase_error = abs(cand_phase - phi_ref_search)
            phase_error = min(phase_error, period_search_px - phase_error)

            if phase_error <= MAX_PHASE_ERROR_PX:
                conf = period_conf * (1.0 - phase_error / MAX_PHASE_ERROR_PX)
                logger.info("Phase match: cand=(%.1f, %.1f), phase_err=%.2f px, conf=%.3f",
                            cand.x, cand.y, phase_error, conf)
                if conf > best_conf:
                    best_conf = conf
                    best_candidate = cand
                break  # Take the highest-scored candidate that phase-matches

    return best_candidate, best_conf
