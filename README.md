# Drift-Sense: AI-Powered Navigation-Error Recovery

**SEMICON India Hackathon 2026 / i4C — Track 2**
Applied Materials Challenge

> **The Problem:** A wafer inspection tool captures a high-resolution reference image of a microscopic chip area (1µm x 1µm). Later, it needs to find that exact same spot inside a much wider, lower-resolution search image (10µm x 10µm) despite extreme noise, stage drift, and distortion.

---

## Quick Start Guide

**1. Setup your environment**
```bash
git clone https://github.com/<YOUR-USERNAME>/drift-sense-winner.git
cd drift-sense-winner
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
```

**2. Generate a Test Dataset**
Create 30 synthetic testing images with real SEM physics (noise, distortion, etc.):
```bash
python generate_dataset.py --num-samples 30 --split test --output-dir ./data --seed 42
```

**3. Run the Processing Engine (Inference)**
Test the algorithm on a single image pair:
```bash
python localize.py --reference data/test/reference/00000.png --search data/test/search/00000.png
```

**4. Evaluate Performance**
Run the pipeline across the entire dataset to generate accuracy graphs and latency metrics:
```bash
python evaluate.py --manifest data/test/manifest.csv --tolerance-px 5 --output-dir ./results
```

---

## How It Works: End-to-End Analysis

### 1. The Data Inputs (`generate_dataset.py` & `src/pipeline.py`)
To test our system, we generate synthetic SEM (Scanning Electron Microscope) images.
- **Reference Image**: A clean, high-resolution 1000x1000 pixel crop of a chip (1 nm/pixel).
- **Search Image**: A wider 1000x1000 pixel image (10 nm/pixel) where the reference is hiding.
- **The Physics Engine**: Before the algorithm sees the images, our generator (`src/sem_imaging.py`) applies 13 real-world physics effects to them, including extreme Poisson shot noise, barrel distortion, charging streaks, and stage rotation.

### 2. The Processing Engine & Preemptive Detection (`localize.py` & `src/matcher/fallback.py`)
When you pass the images into `localize.py`, our dual-engine architecture takes over:

* **Stage 1: Fast-Path Classical Matching (ZNCC)**
  The system first applies Min-Max contrast normalization and runs a fast Zero-mean Normalized Cross-Correlation (ZNCC) search. 
  - *If the image has moderate noise*, ZNCC finds the target in under 90 milliseconds with extreme sub-pixel accuracy.
  
* **Stage 2: Preemptive Failure Detection**
  Rather than failing blindly, the engine evaluates its own confidence. If the ZNCC maximum correlation score drops below a strict threshold (0.35), the system preemptively detects that extreme Poisson noise or missing physical boundaries are causing an aliasing risk.
  
* **Stage 3: Deep Learning Rescue (U-Net)**
  Images that trigger the failure detection are intercepted and routed to a U-Net Neural Network (`src/models/unet.py` trained via `unet-denoising.ipynb`). 
  - The U-Net strips away the noise and restores the microscopic geometry and macroscopic boundaries (like memory mat trenches).
  - The clean images are passed back to the matcher, allowing it to find the true location flawlessly.

### 3. The Output (`evaluate.py`)
The system outputs the exact `(x, y)` coordinate where the reference image is hiding inside the search image. The `evaluate.py` script compares this to the Ground Truth to generate your final metrics, proving our algorithm operates with sub-pixel accuracy in under 500ms!

---

## Performance Highlights

Our algorithm was extensively evaluated across three distinct datasets to map its performance profile:

- **100% Accuracy**: Achieved on **boundary-biased datasets (dose=400)**. When macroscopic boundaries (like memory mat trenches) are present in the image, the U-Net perfectly restores them and entirely resolves periodic ambiguity, giving the classical matcher a massive structural anchor to lock onto.
- **93% Accuracy**: Achieved on **moderately noisy datasets (dose=500)**. The fast-path classical ZNCC operates flawlessly using only min-max contrast normalization, completely bypassing the U-Net for massive latency savings.
- **88% Accuracy**: Achieved on **extreme-noise highly periodic arrays (dose=350)** lacking any macro-boundaries. In these mathematically ambiguous infinite DRAM grids, the U-Net smooths out the noise, and the system gracefully falls back to a Bayesian geometric prior to break ties and guess the center.
- **Latency**: Averages ~180ms per image (well under the 500ms budget limit).

---

## Complete File Structure

```text
drift-sense-winner/
├── README.md                   # You are reading this!
├── requirements.txt            # Required Python packages
├── CITATIONS.md                # Academic physics papers backing our dataset
├── unet-denoising.ipynb        # Kaggle Notebook used to train the U-Net model
│
├── generate_dataset.py         # CLI tool to create synthetic SEM image datasets
├── localize.py                 # MAIN INFERENCE SCRIPT (The core routing engine)
├── evaluate.py                 # Tests the algorithm and generates accuracy graphs
│
├── src/
│   ├── pipeline.py             # Orchestrates the dataset generation
│   ├── sem_imaging.py          # Adds 13 SEM physics effects (noise, distortion)
│   ├── structural_defects.py   # Simulates collapsed patterns
│   ├── presets.py              # Chip architectures (DRAM, FinFET)
│   │
│   ├── patterns/               # Draws the physical shapes of the chips
│   │   ├── dram.py             # Generates DRAM arrays
│   │   ├── finfet.py           # Generates FinFET lines
│   │   └── zones.py            # Adds macro-boundaries (memory mat trenches)
│   │
│   ├── models/
│   │   └── unet.py             # The PyTorch U-Net Neural Network architecture
│   │
│   └── matcher/
│       ├── fallback.py         # The Core Processing Engine (ZNCC + U-Net)
│       ├── coarse_matcher.py   # Basic matching algorithms
│       └── refine.py           # Sub-pixel mathematical refinement
│
├── data/                       # Where your generated datasets are saved
└── results/                    # Where evaluate.py saves its graphs and metrics
```
