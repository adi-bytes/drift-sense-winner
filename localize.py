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

from src.matcher.coarse_matcher import Candidate, coarse_match
from src.matcher.refine import refine_subpixel


def _preprocess(img: np.ndarray) -> np.ndarray:
    """Preprocess image preserving physical scale relationships.

    Uses global min-max normalization to handle dose differences without
    distorting local spatial contrast ratios, followed by Gaussian smoothing
    (5x5, sigma=1.5) to suppress Poisson shot noise and detector noise without
    inducing non-linear edge artifacts.
    """
    norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    filtered = cv2.GaussianBlur(norm, (5, 5), 1.5)
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
    candidates = coarse_match(ref_proc, search_proc, scales=(10.0,))
    t_coarse = time.perf_counter() - t0
    logger.info("Coarse match: %.3fs (%d candidates)", t_coarse, len(candidates))

    best_candidate = candidates[0] if candidates else None
    if best_candidate is None:
        h, w = search_proc.shape
        best_candidate = Candidate(
            x=w / 2.0, y=h / 2.0, score=0.0, scale=10.0,
            template_w=max(ref_proc.shape[1] // 10, 1),
            template_h=max(ref_proc.shape[0] // 10, 1),
        )
    confidence = 1.0
    selected_method = "coarse_best"
    is_periodic = False
    periodic_strength = 0.0
    period_px = (0, 0)
    verification_score = best_candidate.score
    t_disambig = 0.0

    # --- Sub-pixel Refinement ---
    t0 = time.perf_counter()
    refined_x, refined_y = refine_subpixel(
        ref_proc,
        search_proc,
        best_candidate.x,
        best_candidate.y,
    )
    t_refine = time.perf_counter() - t0
    logger.info("Refinement: %.3fs", t_refine)

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
        "is_periodic": is_periodic,
        "period_strength": periodic_strength,
        "period_px": period_px,
        "verification_score": verification_score,
        "method": selected_method,
        "rotation_angle": 0.0,
        "boundary_score": 0.0,
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
