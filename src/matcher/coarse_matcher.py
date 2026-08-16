"""
Multi-scale ZNCC (zero-mean normalized cross-correlation) coarse matcher.

Extends the starter's simple best-match approach to return the TOP-N
candidate locations across multiple scale factors. This is critical for
periodic semiconductor structures (DRAM arrays) where dozens of ZNCC peaks
have nearly identical scores — disambiguation happens downstream.

Uses cv2.matchTemplate with TM_CCOEFF_NORMED, which *is* ZNCC: for each
window position it subtracts the local mean from both template and window,
correlates, and normalizes by their standard deviations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Default scale factors to search — covering -20% to +20% variation
DEFAULT_SCALES: tuple[float, ...] = (10.0,)
DEFAULT_ROTATIONS: tuple[float, ...] = (0.0,)

CENTER_SELECTION_SCALES: tuple[float, ...] = (
    9.0, 9.5, 9.8, 10.0, 10.2, 10.5, 11.0
)

# Maximum number of candidate peaks to return
DEFAULT_MAX_CANDIDATES: int = 50

# Minimum absolute ZNCC score to consider a candidate valid
MIN_ABSOLUTE_SCORE: float = 0.05

# Candidates below this fraction of the best score are discarded
RELATIVE_SCORE_THRESHOLD: float = 0.6

# Minimum distance (in pixels) between two candidate peaks to avoid
# returning near-duplicate detections of the same physical location.
MIN_PEAK_DISTANCE_PX: int = 5


@dataclass
class Candidate:
    """A single candidate match location."""

    x: float  # Center x in search image coordinates
    y: float  # Center y in search image coordinates
    score: float  # ZNCC score in [-1, 1]
    scale: float  # Scale factor that produced this candidate
    template_w: int  # Template width at this scale
    template_h: int  # Template height at this scale
    rotation: float = 0.0  # Rotation angle in degrees


def _find_local_maxima(
    corr_map: np.ndarray,
    min_distance: int = MIN_PEAK_DISTANCE_PX,
    threshold: float = MIN_ABSOLUTE_SCORE,
    max_peaks: int = DEFAULT_MAX_CANDIDATES,
) -> list[tuple[int, int, float]]:
    """Find local maxima in a 2D correlation map.

    Uses morphological dilation to identify peaks, then filters by
    threshold and returns up to max_peaks sorted by score.

    Returns:
        List of (x, y, score) tuples in descending score order.
    """
    # Dilate to find local maxima — a pixel is a local max if it equals
    # its dilated value (i.e., it's the max in its neighborhood).
    kernel_size = 2 * min_distance + 1
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (kernel_size, kernel_size),
    )
    dilated = cv2.dilate(corr_map.astype(np.float32), kernel)
    local_max_mask = (corr_map >= dilated) & (corr_map >= threshold)

    # Extract peak coordinates
    ys, xs = np.where(local_max_mask)
    if len(xs) == 0:
        return []

    scores = corr_map[ys, xs]
    # Sort by descending score
    order = np.argsort(-scores)
    peaks: list[tuple[int, int, float]] = []
    for idx in order:
        if len(peaks) >= max_peaks:
            break
        peaks.append((int(xs[idx]), int(ys[idx]), float(scores[idx])))

    return peaks


def coarse_match(
    reference: np.ndarray,
    search: np.ndarray,
    scales: tuple[float, ...] = DEFAULT_SCALES,
    rotations: tuple[float, ...] = DEFAULT_ROTATIONS,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
    min_peak_distance: int = MIN_PEAK_DISTANCE_PX,
    min_score_threshold: float = MIN_ABSOLUTE_SCORE,
) -> list[Candidate]:
    """Multi-scale ZNCC coarse matcher.

    For each scale factor, the reference image is downsampled to produce a
    template, which is then slid over the search image using ZNCC. Local
    maxima in the correlation maps are collected as candidate locations.

    Args:
        reference: Reference image (grayscale uint8, 1000x1000).
        search: Search image (grayscale uint8, 1000x1000).
        scales: Scale factors to search (reference_size / scale = template_size).
        rotations: Rotation angles to search.
        max_candidates: Maximum total candidates to return.
        min_peak_distance: Minimum pixel distance between candidate peaks.
        min_score_threshold: Minimum ZNCC score to be considered a candidate peak.

    Returns:
        List of Candidate objects sorted by descending ZNCC score.
    """
    all_candidates: list[Candidate] = []

    for scale in scales:
        tw = max(int(round(reference.shape[1] / scale)), 1)
        th = max(int(round(reference.shape[0] / scale)), 1)

        # Template must be smaller than search image
        if tw >= search.shape[1] or th >= search.shape[0]:
            logger.debug(
                "Scale %.1f: template %dx%d too large, skipping", scale, tw, th
            )
            continue

        template_base = cv2.resize(reference, (tw, th), interpolation=cv2.INTER_AREA)

        for rot in rotations:
            if rot != 0.0:
                M = cv2.getRotationMatrix2D((tw / 2.0, th / 2.0), rot, 1.0)
                abs_cos = abs(M[0, 0])
                abs_sin = abs(M[0, 1])
                new_w = int(th * abs_sin + tw * abs_cos)
                new_h = int(th * abs_cos + tw * abs_sin)
                M[0, 2] += new_w / 2.0 - tw / 2.0
                M[1, 2] += new_h / 2.0 - th / 2.0
                # Use CONSTANT border, then crop the valid center
                template = cv2.warpAffine(template_base, M, (new_w, new_h), borderMode=cv2.BORDER_REPLICATE)
                cur_tw, cur_th = new_w, new_h
            else:
                template = template_base
                cur_tw, cur_th = tw, th

            if cur_tw >= search.shape[1] or cur_th >= search.shape[0]:
                continue

            corr_map = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)

            peaks = _find_local_maxima(
                corr_map,
                min_distance=min_peak_distance,
                threshold=min_score_threshold,
                max_peaks=max_candidates,
            )

            for px, py, score in peaks:
                all_candidates.append(
                    Candidate(
                        x=px + cur_tw / 2.0,
                        y=py + cur_th / 2.0,
                        score=score,
                        scale=scale,
                        template_w=cur_tw,
                        template_h=cur_th,
                        rotation=rot,
                    )
                )

    if not all_candidates:
        # Fallback: return center of search image
        logger.warning("No candidates found — returning search image center")
        h, w = search.shape
        return [
            Candidate(
                x=w / 2.0,
                y=h / 2.0,
                score=0.0,
                scale=10.0,
                template_w=100,
                template_h=100,
                rotation=0.0,
            )
        ]

    # Sort by descending score
    all_candidates.sort(key=lambda c: c.score, reverse=True)

    # Apply relative threshold: discard candidates well below the best
    best_score = all_candidates[0].score
    score_cutoff = max(best_score * RELATIVE_SCORE_THRESHOLD, min_score_threshold)
    filtered = [c for c in all_candidates if c.score >= score_cutoff]

    # Deduplicate: merge candidates that are too close together (keep higher score)
    deduped: list[Candidate] = []
    for c in filtered:
        too_close = False
        for existing in deduped:
            dist = np.hypot(c.x - existing.x, c.y - existing.y)
            if dist < min_peak_distance:
                too_close = True
                break
        if not too_close:
            deduped.append(c)
        if len(deduped) >= max_candidates:
            break

    logger.info(
        "Coarse match: %d raw peaks -> %d after threshold -> %d after dedup",
        len(all_candidates),
        len(filtered),
        len(deduped),
    )

    return deduped


def find_center_peak(
    reference: np.ndarray,
    search: np.ndarray,
    scales: tuple[float, ...] = CENTER_SELECTION_SCALES,
    threshold_ratio: float = 0.92,
    min_distance: int = 3,
) -> tuple[Candidate, int]:
    """Find all strong multi-scale peaks and return the one nearest center.

    This implements the problem statement's explicit ambiguity rule. The
    returned count is the number of strong local peaks across all scales and
    is used by ``localize.py`` to report whether the override was triggered.
    """
    h, w = search.shape
    center_x, center_y = w / 2.0, h / 2.0
    best: Candidate | None = None
    best_distance = float("inf")
    unique_centers: list[tuple[float, float]] = []

    for scale in scales:
        tw = max(int(round(reference.shape[1] / scale)), 1)
        th = max(int(round(reference.shape[0] / scale)), 1)
        if tw >= w or th >= h:
            continue
        template = cv2.resize(reference, (tw, th), interpolation=cv2.INTER_AREA)
        corr_map = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
        global_max = float(corr_map.max())
        threshold = global_max * threshold_ratio
        kernel_size = 2 * min_distance + 1
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (kernel_size, kernel_size)
        )
        local_max = cv2.dilate(corr_map, kernel)
        valid = (corr_map >= local_max) & (corr_map >= threshold)
        ys, xs = np.where(valid)
        for px, py in zip(xs, ys):
            score = float(corr_map[py, px])
            cx, cy = float(px + tw / 2.0), float(py + th / 2.0)
            if any(
                np.hypot(cx - ux, cy - uy) < float(min_distance * 2)
                for ux, uy in unique_centers
            ):
                continue
            unique_centers.append((cx, cy))
            distance = float(np.hypot(cx - center_x, cy - center_y))
            if distance < best_distance:
                best_distance = distance
                best = Candidate(
                    x=cx,
                    y=cy,
                    score=score,
                    scale=scale,
                    template_w=tw,
                    template_h=th,
                )

    if best is None:
        return Candidate(
            x=center_x,
            y=center_y,
            score=0.0,
            scale=10.0,
            template_w=max(reference.shape[1] // 10, 1),
            template_h=max(reference.shape[0] // 10, 1),
            rotation=0.0,
        ), 0
    return best, len(unique_centers)

def disambiguate_candidates(
    candidates: list[Candidate],
    search_w: int,
    search_h: int,
    tolerance: float = 0.98
) -> Candidate:
    """Find the best candidate, breaking ties using center proximity.
    
    Groups all candidates whose score is within `tolerance` of the global max score.
    Returns the one whose (x,y) is closest to the image center (search_w/2, search_h/2).
    """
    if not candidates:
        return Candidate(search_w / 2.0, search_h / 2.0, 0.0, 10.0, 100, 100, 0.0)
        
    best_score = max(c.score for c in candidates)
    tied = [c for c in candidates if c.score >= best_score * tolerance]
    
    cx, cy = search_w / 2.0, search_h / 2.0
    return min(tied, key=lambda c: np.hypot(c.x - cx, c.y - cy))
