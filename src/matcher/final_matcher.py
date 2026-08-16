"""Rotation-aware multi-scale matching with boundary verification."""

from __future__ import annotations

import time
from typing import Any

import cv2
import numpy as np
from scipy.ndimage import maximum_filter

from src.matcher.boundary import boundary_score_with_rotation
from src.matcher.rotation import estimate_rotation_between, rotate_image


FINAL_SCALES: tuple[float, ...] = (9.0, 9.5, 9.8, 10.0, 10.2, 10.5, 11.0)


def match_final(
    reference: np.ndarray,
    search: np.ndarray,
    scales: tuple[float, ...] = FINAL_SCALES,
    border_px: int = 5,
) -> dict[str, Any]:
    """Match a reference using rotation correction, ZNCC, and boundary scores."""
    total_start = time.perf_counter()
    rotation_start = time.perf_counter()
    rotation_angle = estimate_rotation_between(reference, search)
    # FFT orientation can be unreliable for nearly isotropic/no-texture images.
    if not np.isfinite(rotation_angle) or abs(rotation_angle) > 10.0:
        rotation_angle = 0.0
    reference_rotated = rotate_image(reference, rotation_angle) if rotation_angle else reference
    rotation_time = time.perf_counter() - rotation_start

    search_h, search_w = search.shape
    candidates: list[dict[str, Any]] = []
    zncc_start = time.perf_counter()
    for scale in scales:
        tw = max(int(round(reference_rotated.shape[1] / scale)), 1)
        th = max(int(round(reference_rotated.shape[0] / scale)), 1)
        if tw >= search_w or th >= search_h:
            continue
        template = cv2.resize(reference_rotated, (tw, th), interpolation=cv2.INTER_AREA)
        result = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
        local_max = maximum_filter(result, size=5) == result
        threshold = float(result.max()) * 0.85
        ys, xs = np.where(local_max & (result >= threshold))
        for y, x in zip(ys, xs):
            candidates.append({
                "x": float(x + tw / 2.0), "y": float(y + th / 2.0),
                "score": float(result[y, x]), "scale": scale,
                "template_w": tw, "template_h": th,
            })
    zncc_time = time.perf_counter() - zncc_start

    if not candidates:
        return {"x": search_w / 2.0, "y": search_h / 2.0, "score": 0.0,
                "rotation_angle": rotation_angle, "boundary_score": 0.0,
                "timings": {"rotation": rotation_time, "zncc": zncc_time,
                            "boundary": 0.0, "total": time.perf_counter() - total_start}}

    candidates.sort(key=lambda item: item["score"], reverse=True)
    boundary_start = time.perf_counter()
    scored: list[dict[str, Any]] = []
    for candidate in candidates[:15]:
        tw, th = int(candidate["template_w"]), int(candidate["template_h"])
        cx, cy = candidate["x"], candidate["y"]
        x0 = max(0, int(round(cx - tw / 2.0)))
        y0 = max(0, int(round(cy - th / 2.0)))
        x1, y1 = min(search_w, x0 + tw), min(search_h, y0 + th)
        patch = search[y0:y1, x0:x1]
        if patch.size == 0:
            continue
        patch_resized = cv2.resize(patch, (tw, th), interpolation=cv2.INTER_AREA)
        template = cv2.resize(reference_rotated, (tw, th), interpolation=cv2.INTER_AREA)
        boundary_score, boundary_angle = boundary_score_with_rotation(template, patch_resized, border_px=border_px)
        positive_boundary = max(0.0, boundary_score)
        candidate["boundary_score"] = boundary_score
        candidate["boundary_angle"] = boundary_angle
        candidate["combined"] = 0.45 * candidate["score"] + 0.45 * positive_boundary + 0.10 * candidate["score"] * positive_boundary
        scored.append(candidate)
    boundary_time = time.perf_counter() - boundary_start
    best = max(scored, key=lambda item: item["combined"]) if scored else candidates[0]
    best["rotation_angle"] = rotation_angle
    best["timings"] = {"rotation": rotation_time, "zncc": zncc_time,
                        "boundary": boundary_time, "total": time.perf_counter() - total_start}
    return best
