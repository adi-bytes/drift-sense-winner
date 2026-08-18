#!/usr/bin/env python3
"""Self-evaluation framework for the Drift-Sense localization pipeline.

Reads a manifest.csv produced by generate_dataset.py, runs the localization
pipeline on each sample, and produces:
  - Confusion matrix at 1-5 px tolerance
  - Overall accuracy, mean/median/max error
  - Per-architecture breakdown (DRAM vs FinFET)
  - Computation time statistics
  - Visual outputs: success cases, failure case, charts

Usage:
    python evaluate.py --manifest data/test/manifest.csv --tolerance-px 5 \
        --output-dir ./results
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
import time
from collections import defaultdict

import cv2
import matplotlib
import numpy as np

from localize import localize

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # pylint: disable=wrong-import-position

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _load_manifest(manifest_path: str) -> list[dict]:
    """Load manifest CSV and return list of row dicts."""
    with open(manifest_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _draw_box(
    img: np.ndarray,
    x: float,
    y: float,
    w: float,
    h: float,
    color: tuple[int, int, int],
    thickness: int = 2,
) -> np.ndarray:
    """Draw a rectangle on a BGR image."""
    x0, y0 = round(x - w / 2), round(y - h / 2)
    x1, y1 = round(x + w / 2), round(y + h / 2)
    cv2.rectangle(img, (x0, y0), (x1, y1), color, thickness)
    return img


def _save_success_grid(
    results: list[dict],
    output_path: str,
    tolerance: float,
    max_cases: int = 9,
) -> None:
    """Save a 3x3 grid of successful match cases."""
    successes = [r for r in results if r["error"] <= tolerance]
    if not successes:
        logger.warning("No successes to display")
        return

    n = min(len(successes), max_cases)
    cols = 3
    rows = (n + cols - 1) // cols

    _fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    if rows == 1:
        axes = [axes] if cols == 1 else axes
    axes_flat = np.array(axes).flatten()

    for i in range(n):
        r = successes[i]
        ax = axes_flat[i]

        search = cv2.imread(r["search_path"], cv2.IMREAD_GRAYSCALE)
        if search is None:
            continue
        search_bgr = cv2.cvtColor(search, cv2.COLOR_GRAY2BGR)

        # Ground truth box (green)
        gt_x, gt_y = r["gt_x"], r["gt_y"]
        gt_w, gt_h = float(r.get("gt_box_w", 100.0)), float(r.get("gt_box_h", 100.0))
        _draw_box(search_bgr, gt_x, gt_y, gt_w, gt_h, (0, 255, 0), 2)

        # Predicted box (red)
        pred_x, pred_y = r["pred_x"], r["pred_y"]
        _draw_box(search_bgr, pred_x, pred_y, gt_w, gt_h, (0, 0, 255), 2)

        search_rgb = cv2.cvtColor(search_bgr, cv2.COLOR_BGR2RGB)
        ax.imshow(search_rgb, cmap="gray")
        ax.set_title(
            f"#{r['id']} {r['architecture']}\nerr={r['error']:.2f}px", fontsize=9
        )
        ax.axis("off")

    for i in range(n, len(axes_flat)):
        axes_flat[i].axis("off")

    plt.suptitle("Success Cases (green=GT, red=predicted)", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def _save_failure_case(
    results: list[dict],
    output_path: str,
    text_path: str,
) -> None:
    """Save the worst failure case with detailed analysis."""
    if not results:
        return

    # Find the worst failure
    worst = max(results, key=lambda r: r["error"])

    if worst["error"] <= 5.0:
        logger.info("No failures (all within 5px) — saving highest-error case")

    _fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Reference image
    ref = cv2.imread(worst["reference_path"], cv2.IMREAD_GRAYSCALE)
    if ref is not None:
        axes[0].imshow(ref, cmap="gray")
        axes[0].set_title("Reference Image\n(1000x1000 @ 1 nm/px)", fontsize=10)
        axes[0].axis("off")

    # Search image with GT and predicted boxes
    search = cv2.imread(worst["search_path"], cv2.IMREAD_GRAYSCALE)
    if search is not None:
        search_bgr = cv2.cvtColor(search, cv2.COLOR_GRAY2BGR)
        gt_w, gt_h = (
            float(worst.get("gt_box_w", 100.0)),
            float(worst.get("gt_box_h", 100.0)),
        )
        _draw_box(search_bgr, worst["gt_x"], worst["gt_y"], gt_w, gt_h, (0, 255, 0), 2)
        _draw_box(
            search_bgr, worst["pred_x"], worst["pred_y"], gt_w, gt_h, (0, 0, 255), 2
        )
        search_rgb = cv2.cvtColor(search_bgr, cv2.COLOR_BGR2RGB)
        axes[1].imshow(search_rgb)
        axes[1].set_title("Search Image\n(green=GT, red=predicted)", fontsize=10)
        axes[1].axis("off")

    # Zoomed comparison
    if search is not None:
        zoom_cx = round(worst["gt_x"])
        zoom_cy = round(worst["gt_y"])
        zoom_r = 80
        y0 = max(zoom_cy - zoom_r, 0)
        y1 = min(zoom_cy + zoom_r, search.shape[0])
        x0 = max(zoom_cx - zoom_r, 0)
        x1 = min(zoom_cx + zoom_r, search.shape[1])
        zoomed = search_rgb[y0:y1, x0:x1]
        axes[2].imshow(zoomed)
        axes[2].set_title(
            f"Zoomed GT Region\nerror={worst['error']:.2f}px", fontsize=10
        )
        axes[2].axis("off")

    plt.suptitle(
        f"Failure Case: #{worst['id']} ({worst['architecture']})\n"
        f"GT=({worst['gt_x']:.1f}, {worst['gt_y']:.1f}), "
        f"Pred=({worst['pred_x']:.1f}, {worst['pred_y']:.1f}), "
        f"Error={worst['error']:.2f}px",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    # Write text analysis
    is_dram = "dram" in worst["architecture"]
    with open(text_path, "w", encoding="utf-8") as f:
        f.write("=" * 72 + "\n")
        f.write("FAILURE CASE ANALYSIS\n")
        f.write("=" * 72 + "\n\n")
        f.write(f"Sample ID: {worst['id']}\n")
        f.write(f"Architecture: {worst['architecture']}\n")
        f.write(f"Ground Truth: ({worst['gt_x']:.2f}, {worst['gt_y']:.2f})\n")
        f.write(f"Predicted:    ({worst['pred_x']:.2f}, {worst['pred_y']:.2f})\n")
        f.write(f"Error:        {worst['error']:.2f} px\n")
        f.write(f"Time:         {worst['time']:.3f} s\n\n")
        f.write("ROOT CAUSE ANALYSIS\n")
        f.write("-" * 40 + "\n")
        if is_dram and worst["error"] > 5.0:
            f.write(
                "This failure occurred in a DRAM array region where the\n"
                "pattern is highly periodic (word-line/bit-line grid repeats\n"
                "every 3-10 pixels at search resolution). The ZNCC matcher\n"
                "produced multiple peaks of nearly equal height at adjacent\n"
                "grid cells. The deterministic disambiguator could not resolve the\n"
                "ambiguity because the local 100x100 patch lacked distinctive\n"
                "features (no mat boundary, no missing/shifted contact via).\n\n"
                "The algorithm selected the peak closest to the search image\n"
                "center, but the true location was at a different grid repeat.\n\n"
            )
            f.write("PROPOSED FIX\n")
            f.write("-" * 40 + "\n")
            f.write(
                "1. Increase macro-context and evaluate candidate locations across\n"
                "   multiple acquisition transforms rather than one local repeat.\n"
                "2. Calibrate the disambiguation weights on a held-out validation\n"
                "   set and report confidence alongside the predicted location.\n"
                "3. If the reference crop is known to contain a mat/strip boundary,\n"
                "   weight boundary-containing candidates higher.\n"
            )
        else:
            f.write(
                f"Error of {worst['error']:.2f} px. The localization was\n"
                f"{'within acceptable bounds.' if worst['error'] <= 5.0 else 'outside tolerance.'}\n"
                "This case represents the maximum error in the test set.\n"
            )


def _save_accuracy_chart(
    errors: list[float],
    output_path: str,
    tolerances: tuple[int, ...] = (1, 2, 3, 4, 5),
) -> None:
    """Bar chart of accuracy at different pixel tolerances."""
    errors_arr = np.array(errors)
    accuracies = [float((errors_arr <= t).mean()) * 100 for t in tolerances]

    _fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        [str(t) for t in tolerances],
        accuracies,
        color=["#2ecc71" if a >= 80 else "#e74c3c" for a in accuracies],
        edgecolor="black",
        linewidth=0.5,
    )

    for bar_patch, acc in zip(bars, accuracies):
        ax.text(
            bar_patch.get_x() + bar_patch.get_width() / 2,
            bar_patch.get_height() + 1,
            f"{acc:.1f}%",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    ax.set_xlabel("Tolerance (pixels)", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Localization Accuracy by Tolerance", fontsize=14)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

def _save_predictions_csv(results: list[dict], output_path: str) -> None:
    """Export all predictions and ground truths to a CSV file."""
    if not results:
        return
    
    fieldnames = [
        "id", "architecture", "error", "gt_x", "gt_y", "pred_x", "pred_y", 
        "time", "confidence_label"
    ]
    
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow(r)


def _save_error_histogram(errors: list[float], output_path: str) -> None:
    """Histogram of prediction errors."""
    _fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(errors, bins=30, color="#3498db", edgecolor="black", alpha=0.8)
    ax.axvline(
        np.median(errors),
        color="red",
        linestyle="--",
        label=f"Median={np.median(errors):.2f}",
    )
    ax.axvline(
        np.mean(errors),
        color="orange",
        linestyle="--",
        label=f"Mean={np.mean(errors):.2f}",
    )
    ax.set_xlabel("Euclidean Error (pixels)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Error Distribution", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def _save_summary(
    results: list[dict],
    output_path: str,
    tolerance: float,
) -> None:
    """Write a text summary of all evaluation metrics."""
    errors = [r["error"] for r in results]
    times = [r["time"] for r in results]
    errors_arr = np.array(errors)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=" * 72 + "\n")
        f.write("DRIFT-SENSE EVALUATION RESULTS\n")
        f.write("=" * 72 + "\n\n")

        f.write(f"Total samples:    {len(results)}\n")
        f.write(f"Tolerance:        {tolerance} px\n\n")

        f.write("CONFUSION MATRIX (Tolerance vs Matches)\n")
        f.write("-" * 55 + "\n")
        f.write(f"{'Tolerance':<12} | {'Match (TP)':<12} | {'Mismatch (FP)':<15} | {'Accuracy':<10}\n")
        f.write("-" * 55 + "\n")
        for tol in [1, 2, 3, 4, 5]:
            matches = int((errors_arr <= tol).sum())
            mismatches = len(errors) - matches
            acc = (matches / len(errors)) * 100
            f.write(f"<= {tol}px{'':<5} | {matches:<12} | {mismatches:<15} | {acc:.1f}%\n")

        f.write("\nERROR STATISTICS\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Mean:   {np.mean(errors):8.3f} px\n")
        f.write(f"  Median: {np.median(errors):8.3f} px\n")
        f.write(f"  Std:    {np.std(errors):8.3f} px\n")
        f.write(f"  Max:    {np.max(errors):8.3f} px\n")
        f.write(f"  Min:    {np.min(errors):8.3f} px\n")
        confidences = np.asarray([r.get("confidence", 0.0) for r in results])
        f.write(f"  Catastrophic >50px: {int((errors_arr > 50).sum())}\n")
        f.write("\nCONFIDENCE STATISTICS\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Mean:   {np.mean(confidences):8.3f}\n")
        f.write(f"  Min:    {np.min(confidences):8.3f}\n")
        f.write(f"  <0.6:   {int((confidences < 0.6).sum())}/{len(confidences)}\n")
        verification = np.asarray([r.get("verification_score", 0.0) for r in results])
        f.write("\nPERIODICITY BREAKDOWN\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Mean verification score: {np.mean(verification):.3f}\n")

        f.write("\nPREEMPTIVE DIAGNOSTIC ENGINE\n")
        f.write("-" * 40 + "\n")
        labels = [r.get("confidence_label", "UNKNOWN") for r in results]
        unique_labels = set(labels)
        for label in unique_labels:
            mask = np.array([lbl == label for lbl in labels])
            count = int(mask.sum())
            accuracy = float((errors_arr[mask] <= tolerance).mean()) if count else 0.0
            f.write(f"  {label:30s}: n={count:3d}, acc@{tolerance:g}px={accuracy * 100:5.1f}%\n")

        f.write("\nTIMING STATISTICS\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Mean:   {np.mean(times):8.3f} s\n")
        f.write(f"  Std:    {np.std(times):8.3f} s\n")
        f.write(f"  Min:    {np.min(times):8.3f} s\n")
        f.write(f"  Max:    {np.max(times):8.3f} s\n")

        # Per-architecture breakdown
        f.write("\nPER-ARCHITECTURE BREAKDOWN\n")
        f.write("-" * 40 + "\n")
        arch_groups: dict[str, list[float]] = defaultdict(list)
        for r in results:
            kind = "DRAM" if "dram" in r["architecture"] else "FinFET"
            arch_groups[kind].append(r["error"])

        for arch, arch_errors in sorted(arch_groups.items()):
            arr = np.array(arch_errors)
            acc5 = float((arr <= 5.0).mean()) * 100
            acc2 = float((arr <= 2.0).mean()) * 100
            f.write(
                f"  {arch:8s}: n={len(arr):3d}, "
                f"acc@5px={acc5:5.1f}%, acc@2px={acc2:5.1f}%, "
                f"mean_err={arr.mean():.2f}px, median={np.median(arr):.2f}px\n"
            )

        f.write("\n")


def main() -> None:
    """Run evaluation on a generated dataset."""
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--manifest",
        required=True,
        help="Path to manifest.csv from generate_dataset.py",
    )
    p.add_argument(
        "--tolerance-px", type=float, default=5.0, help="Tolerance threshold in pixels"
    )
    p.add_argument(
        "--output-dir", default="./results", help="Directory for evaluation outputs"
    )
    p.add_argument(
        "--optical",
        action="store_true",
        help="Run evaluation using the isolated Optical RGB pipeline",
    )
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    rows = _load_manifest(args.manifest)
    print(f"Evaluating {len(rows)} samples...")

    if not rows:
        print(f"Error: Manifest file {args.manifest} is empty. Please wait for the dataset generation to complete or generate a new dataset.", file=sys.stderr)
        sys.exit(1)

    results: list[dict] = []
    manifest_dir = os.path.dirname(os.path.abspath(args.manifest))

    for i, row in enumerate(rows):
        ref_path = os.path.normpath(os.path.join(manifest_dir, row["reference_path"].replace("\\", "/")))
        search_path = os.path.normpath(os.path.join(manifest_dir, row["search_path"].replace("\\", "/")))
        gt_x = float(row["gt_x"])
        gt_y = float(row["gt_y"])
        gt_box_w = float(row.get("gt_box_w", 100.0))
        gt_box_h = float(row.get("gt_box_h", 100.0))
        architecture = row["architecture"]
        sample_id = row["id"]

        t0 = time.perf_counter()
        failure_reason: str | None = None
        try:
            if not os.path.isfile(ref_path):
                raise FileNotFoundError(ref_path)
            if not os.path.isfile(search_path):
                raise FileNotFoundError(search_path)
            # --- Run Localization ---
            # We explicitly pass return_confidence and return_diagnostics for robust error tracking
            result = localize(
                ref_path, search_path,
                return_confidence=True, return_diagnostics=True,
                optical=args.optical
            )
            # Unpack (localize returns either 2, 3, or 4 tuple based on flags)
            # Since we set both flags, it returns 4 items
            pred_x, pred_y, confidence, diagnostics = result
        except FileNotFoundError as e:
            failure_reason = "missing_file"
            print(f"  [{i + 1}/{len(rows)}] FILE ERROR on sample {sample_id}: {e}", file=sys.stderr)
            pred_x, pred_y = 500.0, 500.0
            confidence = 0.0
            diagnostics = {"period_strength": 0.0}
        except cv2.error as e:
            failure_reason = "opencv_error"
            print(f"  [{i + 1}/{len(rows)}] OpenCV ERROR on sample {sample_id}: {e}", file=sys.stderr)
            pred_x, pred_y = 500.0, 500.0
            confidence = 0.0
            diagnostics = {"period_strength": 0.0}
        except (ValueError, RuntimeError, OSError) as e:
            failure_reason = type(e).__name__
            print(f"  [{i + 1}/{len(rows)}] PIPELINE ERROR on sample {sample_id}: {e}", file=sys.stderr)
            pred_x, pred_y = 500.0, 500.0
            confidence = 0.0
            diagnostics = {"period_strength": 0.0}
        except Exception as e:  # Defensive boundary for one bad sample.
            failure_reason = f"unexpected:{type(e).__name__}"
            print(
                f"  [{i + 1}/{len(rows)}] UNEXPECTED ERROR on sample {sample_id}: {e}",
                file=sys.stderr,
            )
            pred_x, pred_y = 500.0, 500.0
            confidence = 0.0
            diagnostics = {"period_strength": 0.0}

        elapsed = time.perf_counter() - t0
        error = float(np.hypot(pred_x - gt_x, pred_y - gt_y))

        results.append(
            {
                "id": sample_id,
                "architecture": architecture,
                "gt_x": gt_x,
                "gt_y": gt_y,
                "gt_box_w": gt_box_w,
                "gt_box_h": gt_box_h,
                "pred_x": pred_x,
                "pred_y": pred_y,
                "error": error,
                "time": elapsed,
                "reference_path": ref_path,
                "search_path": search_path,
                "failure_reason": failure_reason,
                "confidence": confidence,
                "period_strength": diagnostics.get("period_strength", 0.0),
                "verification_score": diagnostics.get("verification_score", 0.0),
                "confidence_label": diagnostics.get("confidence_label", "UNKNOWN"),
                "max_zncc_score": diagnostics.get("max_zncc_score", 0.0),
                "aliasing_ratio": diagnostics.get("aliasing_ratio", 0.0),
            }
        )

        status = "[OK]" if error <= args.tolerance_px else "[FAIL]"
        conf_label = diagnostics.get("confidence_label", "UNKNOWN")
        print(
            f"  [{i + 1}/{len(rows)}] {status} {architecture} "
            f"err={error:.2f}px time={elapsed:.2f}s [{conf_label}]"
        )

    # Generate outputs
    print("\nGenerating visual outputs...")

    _save_success_grid(
        results,
        os.path.join(args.output_dir, "success_cases.png"),
        args.tolerance_px,
    )
    _save_failure_case(
        results,
        os.path.join(args.output_dir, "failure_case.png"),
        os.path.join(args.output_dir, "failure_analysis.txt"),
    )
    _save_accuracy_chart(
        [r["error"] for r in results],
        os.path.join(args.output_dir, "accuracy_by_tolerance.png"),
    )
    _save_error_histogram(
        [r["error"] for r in results],
        os.path.join(args.output_dir, "error_distribution.png"),
    )
    _save_summary(
        results, os.path.join(args.output_dir, "results_summary.txt"), args.tolerance_px
    )
    _save_predictions_csv(
        results, os.path.join(args.output_dir, "predictions.csv")
    )

    # Print summary to stdout
    errors = [r["error"] for r in results]
    errors_arr = np.array(errors)
    print(f"\n{'=' * 50}")
    print(f"RESULTS SUMMARY ({len(results)} samples)")
    print(f"{'=' * 50}")
    print("\nCONFUSION MATRIX (Matches vs Mismatches)")
    print("-" * 55)
    print(f"{'Tolerance':<12} | {'Match (TP)':<12} | {'Mismatch (FP)':<15} | {'Accuracy':<10}")
    print("-" * 55)
    for tol in [1, 2, 3, 4, 5]:
        matches = int((errors_arr <= tol).sum())
        mismatches = len(errors) - matches
        acc = (matches / len(errors)) * 100
        print(f"<= {tol}px{'':<5} | {matches:<12} | {mismatches:<15} | {acc:.1f}%")
    print(f"  Mean error:     {np.mean(errors):.3f} px")
    print(f"  Median error:   {np.median(errors):.3f} px")
    print(f"  Mean time:      {np.mean([r['time'] for r in results]):.3f} s")
    print(f"\nOutputs saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
