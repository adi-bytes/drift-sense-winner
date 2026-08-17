#!/usr/bin/env python3
"""Main inference script for Drift-Sense localization.

THIS IS THE SCRIPT JUDGES RUN DIRECTLY.

Usage:
    python localize.py --reference path/to/ref.png --search path/to/search.png

Output (to stdout, exactly):
    x,y

Where x and y are floats giving the predicted center of the reference
pattern inside the search image, in search-image pixel coordinates.

Pipeline:
  1. Load images (grayscale)
  2. Preprocess: global min-max normalization + bilateral denoising
  3. Multi-scale ZNCC coarse match -> top-N candidates
  4. Multi-peak center-proximity selection for ambiguous matches
  5. Sub-pixel refinement -> final (x, y)
"""

from __future__ import annotations

import argparse
import logging
import sys
import time

import cv2
import numpy as np

from src.matcher.coarse_matcher import Candidate, coarse_match, CENTER_SELECTION_SCALES
from src.matcher.refine import refine_location
from src.matcher.strip_anchor import strip_anchor_match


def _preprocess(img: np.ndarray) -> np.ndarray:
    """Preprocess image for cross-scale ZNCC matching.

    CLAHE is explicitly avoided here: it uses tile-based local normalization,
    which applies different non-linear transforms to the reference (1nm/px tiles)
    and the search (10nm/px tiles), completely destroying cross-image correlation.

    Instead we use:
    1. Global min-max normalization (dose-invariant, preserves relative contrast)
    2. Bilateral filter (edge-preserving denoising, better than Gaussian at high noise)
    REF: Tomasi & Manduchi (1998) -- bilateral filter standard for SEM denoising.
    """
    norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    # Bilateral: d=9, sigmaColor=30, sigmaSpace=15
    # Stronger than original Gaussian(5,5) but edge-preserving for noisy Level 4-6 images
    filtered = cv2.bilateralFilter(norm, d=7, sigmaColor=25, sigmaSpace=10)
    return filtered


def localize(
    reference_path: str,
    search_path: str,
    verbose: bool = False,
    return_confidence: bool = False,
    return_diagnostics: bool = False,
) -> tuple[float, float] | tuple[float, float, float] | tuple[float, float, float, dict]:
    """Run the full localization pipeline.

    Args:
        reference_path: Path to the reference image (1000x1000 @ 1 nm/px).
        search_path: Path to the search image (1000x1000 @ 10 nm/px).
        verbose: If True, log detailed timing and diagnostics.

    Returns:
        (x, y) predicted center in search image coordinates.
    """
    if verbose:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )

    logger = logging.getLogger(__name__)
    t_total = time.perf_counter()

    # --- Load ---
    t0 = time.perf_counter()
    reference = cv2.imread(reference_path, cv2.IMREAD_GRAYSCALE)
    search = cv2.imread(search_path, cv2.IMREAD_GRAYSCALE)

    if reference is None:
        print(
            f"Error: Could not read reference image: {reference_path}", file=sys.stderr
        )
        sys.exit(1)
    if search is None:
        print(f"Error: Could not read search image: {search_path}", file=sys.stderr)
        sys.exit(1)

    t_load = time.perf_counter() - t0
    logger.info(
        "Load: %.3fs (ref=%s, search=%s)", t_load, reference.shape, search.shape
    )

    # --- Preprocess ---
    t0 = time.perf_counter()
    ref_proc = _preprocess(reference)
    search_proc = _preprocess(search)
    t_preprocess = time.perf_counter() - t0
    logger.info("Preprocess: %.3fs", t_preprocess)

    # --- Coarse Match ---
    t0 = time.perf_counter()
    candidates = coarse_match(ref_proc, search_proc)
    t_coarse = time.perf_counter() - t0
    logger.info("Coarse match: %.3fs (%d candidates)", t_coarse, len(candidates))

    # --- Strategy 1: Strip Anchor (ONLY when ZNCC is ambiguous) ---
    # Strip anchor is expensive in terms of false positives when ZNCC is already confident.
    # Only run it when ZNCC has multiple high-scoring candidates (aliasing risk).
    initial_max = candidates[0].score if candidates else 0.0
    aliasing_ratio = (candidates[1].score / initial_max) if len(candidates) > 1 and initial_max > 0 else 0.0
    zncc_is_unambiguous = initial_max > 0.45 and aliasing_ratio < 0.88
    
    strip_result = None
    if not zncc_is_unambiguous and initial_max >= 0.35:
        # ZNCC is ambiguous — try strip anchor to resolve it
        t0 = time.perf_counter()
        strip_result = strip_anchor_match(reference, search)
        t_strip = time.perf_counter() - t0
        if strip_result is not None:
            logger.info(
                "Strip anchor fired (conf=%.3f, dir=%s) → (%.1f, %.1f) [%.3fs]",
                strip_result.confidence, strip_result.direction,
                strip_result.x, strip_result.y, t_strip
            )
    
    # Adaptive Cascade Router: heavy fallback when ZNCC finds nothing
    if initial_max < 0.35:
        logger.warning(
            "WARNING_SEVERE_NOISE_OR_OOD: Triggering Heavy Fallback Pipeline (score=%.3f)",
            initial_max
        )
        t_fb = time.perf_counter()
        from src.matcher.fallback import heavy_match
        candidates, ref_proc, search_proc = heavy_match(reference, search)
        logger.info("Fallback execution time: %.3fs", time.perf_counter() - t_fb)

    # --- Disambiguation & Refinement ---
    t0 = time.perf_counter()
    
    # Defaults for diagnostics
    periodic_strength = 0.0
    period_px = (0, 0)
    selected_method = "coarse_best"
    verification_score = 0.0
    t_disambig = 0.0
    
    if not candidates:
        h, w = search_proc.shape
        refined_x, refined_y = w / 2.0, h / 2.0
        confidence = 0.0
    elif strip_result is not None and strip_result.confidence >= 0.15:
        # Strategy 1 wins: strip anchor gives us an unambiguous location
        # Guard: only trust strip anchor when combined confidence is meaningful
        logger.info("Using strip anchor result (conf=%.3f)", strip_result.confidence)
        from src.matcher.coarse_matcher import Candidate as Cand
        anchor_cand = Cand(
            x=strip_result.x, y=strip_result.y, score=strip_result.confidence,
            scale=10.0, template_w=100, template_h=100, rotation=0.0
        )
        rx, ry, rscore = refine_location(ref_proc, search_proc, anchor_cand)
        refined_x, refined_y = rx, ry
        confidence = float(rscore)
        selected_method = "strip_anchor"
    else:
        # No strip anchor available — use multi-scale ZNCC voting to pick the most
        # consistent candidate across multiple template scales. The correct peak is the
        # one that is stably top-ranked across scales; aliasing peaks tend to drop out
        # at scales that don't match the template exactly.
        from src.matcher.coarse_matcher import coarse_match as _cm
        vote_counts: dict[tuple[int,int], int] = {}
        for sc in CENTER_SELECTION_SCALES:
            vc = _cm(ref_proc, search_proc, scales=(sc,), max_candidates=5)
            for c in vc[:3]:
                key = (int(round(c.x / 5) * 5), int(round(c.y / 5) * 5))
                vote_counts[key] = vote_counts.get(key, 0) + 1
        
        # Find candidate bin with highest vote count
        best_voted_key = max(vote_counts, key=vote_counts.get) if vote_counts else None
        
        if best_voted_key is not None and vote_counts[best_voted_key] >= 3:
            # Strong multi-scale consensus — find the actual candidate closest to this bin
            bvx, bvy = best_voted_key
            voted_cand = min(candidates, key=lambda c: (int(round(c.x/5)*5) - bvx)**2 + (int(round(c.y/5)*5) - bvy)**2)
            rx, ry, rscore = refine_location(ref_proc, search_proc, voted_cand)
            refined_x, refined_y = rx, ry
            confidence = float(rscore)
            selected_method = "multiscale_vote"
            logger.info("Multi-scale vote consensus: (%.1f, %.1f) with %d votes", rx, ry, vote_counts[best_voted_key])
        else:
            # Final fallback: center-proximity tie-breaking
            refined_results = []
            for c in candidates:
                rx, ry, rscore = refine_location(ref_proc, search_proc, c)
                refined_results.append((rx, ry, rscore, c))
            max_rscore = max(r[2] for r in refined_results)
            tied_refined = [r for r in refined_results if r[2] >= max_rscore - 0.001]
            h, w = search_proc.shape
            cx, cy = w / 2.0, h / 2.0
            final_choice = min(tied_refined, key=lambda r: np.hypot(r[0] - cx, r[1] - cy))
            refined_x, refined_y = final_choice[0], final_choice[1]
            confidence = float(final_choice[2])
            selected_method = "center_proximity"
        
        # Diagnostic Engine Extraction
        max_score = candidates[0].score if candidates else 0.0
        aliasing_ratio = (candidates[1].score / max_score) if len(candidates) > 1 and max_score > 0 else 0.0
        
        # Determine Preemptive Confidence Label
        if max_score < 0.35:
            confidence_label = "WARNING_SEVERE_NOISE_OR_OOD"
        elif aliasing_ratio > 0.95:
            confidence_label = "WARNING_ALIASING_RISK"
        else:
            confidence_label = "HIGH_CONFIDENCE"
        
    t_refine = time.perf_counter() - t0
    logger.info("Refinement & Disambiguation: %.3fs", t_refine)

    t_total_elapsed = time.perf_counter() - t_total
    logger.info(
        "Total: %.3fs | Load=%.3fs Preprocess=%.3fs Coarse=%.3fs "
        "Disambig=%.3fs Refine=%.3fs",
        t_total_elapsed,
        t_load,
        t_preprocess,
        t_coarse,
        t_disambig,
        t_refine,
    )

    diagnostics = {
        "period_strength": periodic_strength,
        "period_px": period_px,
        "verification_score": verification_score,
        "method": selected_method,
        "max_zncc_score": max_score if 'max_score' in locals() else 0.0,
        "aliasing_ratio": aliasing_ratio if 'aliasing_ratio' in locals() else 0.0,
        "confidence_label": confidence_label if 'confidence_label' in locals() else "FAILURE_NO_CANDIDATES",
        "timings": {},
    }
    if return_diagnostics:
        return refined_x, refined_y, confidence, diagnostics
    if return_confidence:
        return refined_x, refined_y, confidence
    return refined_x, refined_y


def main() -> None:
    """CLI entry point — prints 'x,y' to stdout."""
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--reference",
        required=True,
        help="Path to reference image (1000x1000 @ 1 nm/px)",
    )
    p.add_argument(
        "--search", required=True, help="Path to search image (1000x1000 @ 10 nm/px)"
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Print timing and diagnostic info to stderr",
    )
    args = p.parse_args()

    x, y = localize(args.reference, args.search, verbose=args.verbose)
    print(f"{x:.2f},{y:.2f}")


if __name__ == "__main__":
    main()
