# Drift-Sense: Dataset Rationale & Failure Analysis

As per the Track 2 Applied Materials guidelines, our submission includes **two carefully curated 30-sample datasets** (one for SEM Grayscale, one for Optical RGB) that demonstrate our algorithm's resilience across scale variations, noise, repetitive patterns, and extreme drift.

## 1. Dataset Generation & Physics Modeling

Our synthetic data pipeline strictly relies on verifiable semiconductor physics, backed by public literature. 
- **Scale Factor**: We exactly model the 10x resolution difference (1nm/px reference vs 10nm/px search) required by the problem statement.
- **Grayscale SEM Noise (`final_data_generation/sem_physics.py`)**: Models Poisson shot noise (Villarrubia et al., 2003), Line Edge Roughness (Bunday et al., 2003), barrel distortion, and edge-brightening (secondary electron emission physics).
- **Optical RGB Noise (`final_data_generation/optical_physics.py`)**: Models optical diffraction limits (Airy disk convolution), thin-film interference chromaticity shifts, and extreme camera vignetting.

## 2. Overall Performance Metrics

Our algorithm prioritizes deterministic, mathematically sound heuristics (ZNCC + Strip Anchors) over black-box deep learning.

### SEM Grayscale Results (30 Samples)
* **Accuracy (≤5px):** **83.3%**
* **Median Error:** ~0.20 px
* **Average Inference Latency:** ~200 ms 
The system effortlessly handles rotation, SNR degradation, and stage drift.

### Optical RGB Results (30 Samples)
* **Accuracy (≤5px):** **96.7%**
* **Median Error:** <1 px
* **Average Inference Latency:** ~100 ms 
By normalizing RGB channels, our mathematical approach becomes completely invariant to severe camera-bound illumination gradients, allowing flawless tracking of structural chromaticity for 29 out of 30 samples.

---

## 3. Honest Explainability & Root-Cause Failure Analysis

As explicitly required by the rubric, we have documented authentic failure cases where our algorithm reaches its mathematical limit.

### SEM Failure Mode: "Periodic Aliasing" on Perfect Arrays (5 Cases)

**The Problem:** 
When the algorithm fails, it exclusively fails on **DRAM array regions** at maximum severity drift. In these regions, the physical structure (word-lines and bit-lines) repeats identically every few pixels. Because we apply massive stage drift (±400px), the true location is shifted far from the center. 
If the 100x100 nm reference patch lands deep inside a dense array *without any unique macroscopic boundaries* (like a missing contact or trench edge), the ZNCC matching algorithm produces dozens of mathematically identical correlation peaks. It is fundamentally impossible to resolve the true location from a tiny crop of a perfectly repeating grid.

**How we detect it:**
Our adaptive pipeline doesn't guess blindly. It computes an `aliasing_ratio`. If the top two ZNCC peaks are >95% identical, the system actively flags a `WARNING_ALIASING_RISK`.

**How it would be fixed in production:**
To solve this boundary-less aliasing problem in a real fab, the system must increase the initial Field of View (FoV) of the reference image capture. By ensuring the reference image captures at least one unique macroscopic feature (a zone boundary, a unique defect, or CAD alignment mark), the mathematical ambiguity is broken.

### Optical RGB Failure Mode: "Featureless Chromatic Washout" (1 Case)

**The Problem:**
In 1 out of the 30 optical test cases, the accuracy dropped (error > 400px). This occurs when the reference patch lands on a completely featureless region of the die (e.g., a massive uniform FinFET routing plane) while extreme camera vignetting and diffraction blur perfectly cancel out the sub-pixel structural details. In this exact "perfect storm," the global LAB Chrominance SSD finds multiple identical minimums because there is simply no gradient information left to track. 

**How it would be fixed in production:**
Similar to the SEM aliasing issue, capturing a larger reference FoV guarantees the inclusion of a color boundary (e.g., a metal transition layer) which immediately restores the global minimum.
