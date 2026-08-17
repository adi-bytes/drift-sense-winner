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
# Standard generator (original, now archived)
python archive/generate_dataset.py --num-samples 30 --split test --output-dir ./data --seed 42

# Upgraded generator with physics severity curriculum (recommended)
python3 -m final_data_generation.run --num-samples 30 --severity-level 2 --output-dir ./data --seed 42
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

### 1. The Data Inputs (`archive/generate_dataset.py` & `final_data_generation/`)
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

### 1. Final Upgraded Pipeline (`final_data_generation`)
Evaluated on strictly spec-compliant, randomized datasets incorporating Line Edge Roughness (LER), Smooth Stage Drift, and Sidewall gradients:

| Dataset | Severity | Accuracy @1px | Mean Error | Mean Latency |
|---|---|---|---|---|
| Upgraded generator (moderate noise, drift, LER) | Level 2 | **100.0%** | 0.111 px | 193 ms |
| Original generator (standard noise) | — | 100.0% | 0.09 px | 180 ms |
| Upgraded generator (extreme drift + low dose) | Level 6 | ~73% (honest failure) | — | 195 ms |

- **Fast path (ZNCC)**: Resolves all standard cases in under 200 ms with sub-pixel accuracy.
- **Adaptive fallback (U-Net)**: Activated when the ZNCC confidence score drops below 0.35, restoring severely degraded images.
- **Honest failure case**: Level 6 extreme-severity samples (600 dose, 5 px drift amplitude, 4 nm LER) demonstrate the mathematical limit of periodic aliasing in pure DRAM arrays without macro-boundaries — exactly as the spec requires.
- **Latency**: Averages ~193 ms per image (well under the 500 ms budget limit).

### 2. Initial Development & Ablation Studies (Legacy Generator)
During initial testing with the baseline generator, our algorithm's dual-engine architecture demonstrated its routing capabilities:

- **100% Accuracy**: Achieved on **boundary-biased datasets (dose=400)**. When macroscopic boundaries (like memory mat trenches) are present in the image, the U-Net perfectly restores them and entirely resolves periodic ambiguity, giving the classical matcher a massive structural anchor to lock onto.
- **93% Accuracy**: Achieved on **moderately noisy datasets (dose=500)**. The fast-path classical ZNCC operates flawlessly using only min-max contrast normalization, completely bypassing the U-Net for massive latency savings.
- **88% Accuracy**: Achieved on **extreme-noise highly periodic arrays (dose=350)** lacking any macro-boundaries. In these mathematically ambiguous infinite DRAM grids, the U-Net smooths out the noise, allowing the system to lock onto the correct periodicity.
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
├── archive/                    # Archived baseline dataset generator (MVP)
│   ├── generate_dataset.py     
│   ├── pipeline.py             
│   ├── sem_imaging.py          
│   ├── presets.py              
│   └── patterns/               
│
├── localize.py                 # MAIN INFERENCE SCRIPT (The core routing engine)
├── evaluate.py                 # Tests the algorithm and generates accuracy graphs
│
├── final_data_generation/      # Upgraded physics-backed dataset generator
│   ├── run.py                  # CLI entry point (python3 -m final_data_generation.run)
│   ├── presets.py              # Architecture configs + severity curriculum (levels 0-6)
│   ├── geometry.py             # DRAM/FinFET/zones with LER, sidewall, material SE gains
│   └── sem_physics.py          # Upgraded SEM engine: smooth drift, correlated noise
│
├── src/
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
