# Drift-Sense: AI-Powered Navigation-Error Recovery for Wafer Inspection Tools

**SEMICON India Hackathon 2026 / i4C — Track 2**
Applied Materials Challenge: Drift-Sense

> Given a high-resolution Reference SEM image (1000×1000 px @ 1 nm/px) and a
> 10× lower-magnification Search image (1000×1000 px @ 10 nm/px), find the
> center (x, y) of the reference pattern inside the search image.

---

## Quick Start

```bash
# 1. Clone and setup
git clone <repo-url> drift-sense-winner
cd drift-sense-winner
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt

# 2. Generate test dataset (30 samples)
python generate_dataset.py --num-samples 30 --split test --output-dir ./data --seed 42

# 3. Run inference (THE COMMAND JUDGES RUN)
python localize.py --reference data/test/reference/00000.png --search data/test/search/00000.png

# 4. Evaluate on full dataset
python evaluate.py --manifest data/test/manifest.csv --tolerance-px 5 --output-dir ./results
```

---

## Problem Statement

A wafer inspection tool captures a high-res reference image of a die region.
Later, it must return to the exact same physical spot at 10× lower
magnification. Due to stage drift, it lands off-target. The algorithm must
recover the reference location inside the wider search field.

| Property | Reference | Search |
|---|---|---|
| Resolution | 1000×1000 px | 1000×1000 px |
| Pixel size | 1 nm/px | 10 nm/px |
| FOV | 1 μm × 1 μm | 10 μm × 10 μm |
| Reference footprint | — | ~100×100 px |

---

## Directory Structure

```
drift-sense-winner/
├── README.md                   # This file
├── requirements.txt            # Dependencies
├── CITATIONS.md                # Every parameter mapped to 2-3 papers
├── generate_dataset.py         # CLI dataset generator
├── localize.py                 # ★ MAIN INFERENCE SCRIPT ★
├── evaluate.py                 # Self-evaluation framework
├── src/
│   ├── pipeline.py             # Sample generation orchestrator
│   ├── sem_imaging.py          # SEM physics engine (13 effects)
│   ├── structural_defects.py   # Pattern collapse model
│   ├── presets.py              # 12 architecture presets
│   ├── patterns/
│   │   ├── dram.py             # 6F² DRAM array generator
│   │   ├── finfet.py           # FinFET array generator
│   │   └── zones.py            # Multi-region die layout composer
│   └── matcher/
│       ├── coarse_matcher.py   # ZNCC coarse matcher (local maxima extraction)
│       └── refine.py           # Sub-pixel refinement (parabolic peak fit)
├── models/                     # Model weights (if needed)
├── data/                       # Generated datasets
├── results/                    # Evaluation outputs
└── tests/
    └── test_pipeline.py        # Unit tests
```

---

## Key Design Decisions

### 1. Physics-Grounded Fast Classical Pipeline

The localization engine uses a robust, deterministic two-stage approach:

1. **Preprocess**: Global min-max contrast normalization followed by calibrated Gaussian filtering (5×5, σ=1.5). This suppresses Poisson shot noise and high-frequency detector noise without introducing non-linear edge artifacts.
2. **Coarse Matching**: Single-scale (10.0×) Zero-mean Normalized Cross-Correlation (ZNCC via `cv2.matchTemplate(TM_CCOEFF_NORMED)`) to find the global optimum across the 1000×1000 search field.
3. **Sub-pixel Refinement**: Local search window NCC with 1D parabolic peak interpolation along both orthogonal axes, achieving <0.6 px median accuracy.

**Why this design?** The physical zoom ratio is fixed and known (exactly 10×). By matching the preprocessing to the underlying SEM imaging physics (low-pass filtering to counter Poisson variance) and using the full template ZNCC, the matcher avoids spurious multi-scale candidates while running in under 90ms on standard CPU.

### 2. Physics-Grounded Dataset Generator

The SEM imaging engine models **13 physical effects** with literature
citations for each:

| Effect | Ref → Search | Citation Key |
|---|---|---|
| Beam PSF blur | Both | Reimer 1998 |
| Edge brightening | Both | Seiler 1983 |
| Shot noise (Poisson) | Both (high/low dose) | Kockentiedt 2013 |
| Detector noise (Gaussian) | Both | Goldstein 2018 |
| Speckle (multiplicative) | Both | Müllerová 2003 |
| Salt & pepper (impulse) | Both | Goldstein 2018 |
| Raster drift + jitter | Light → Heavy | Sutton 2007 |
| Barrel distortion | Light → Full | Reimer 1998 |
| Vignetting | Light → Full | Goldstein 2018 |
| Gamma nonlinearity | Both | Goldstein 2018 |
| Charging streaks | Both | Cazaux 1995 |
| Stage rotation | Search only | Postek 2004 |
| B/C jitter | Light → Full | Postek 2004 |

### 3. DRAM as Primary Challenge

DRAM arrays are the harder matching problem due to extreme periodicity
(every 3–10 px at search resolution). FinFET structures have more
distinctive features and match more easily.

---

## Performance Targets

| Metric | Target |
|---|---|
| Accuracy @5px | ≥85% |
| Accuracy @2px | ≥60% |
| Inference time | <500ms/pair (CPU) |
| Test cases | ≥30 |

---

## Evaluation Outputs

After running `evaluate.py`, the `results/` directory contains:

- `results_summary.txt` — Full text metrics
- `accuracy_by_tolerance.png` — Bar chart at 1–5px
- `error_distribution.png` — Error histogram
- `success_cases.png` — 3×3 grid of good matches
- `failure_case.png` — Worst failure with annotation
- `failure_analysis.txt` — Root-cause explanation + proposed fix

---

## Citation Summary

See [CITATIONS.md](CITATIONS.md) for the complete reference list. Every
augmentation parameter is backed by 2–3 sources from:

- Reimer, *Scanning Electron Microscopy*, Springer 1998
- Goldstein et al., *SEM and X-ray Microanalysis*, Springer 2018
- Kockentiedt et al., SPIE 2013
- Seiler, *J. Applied Physics*, 1983
- Postek & Vladár, *Scanning*, 2004
- Keeth et al., *DRAM Circuit Design*, Wiley 2007
- Iwaki et al., euspen ICE16, 2016

---

## License

Submission for SEMICON India Hackathon 2026. Based on the
[drift-sense-synthetic-data](https://huggingface.co/spaces/aayushraina21/drift-sense-synthetic-data)
starter repository.
