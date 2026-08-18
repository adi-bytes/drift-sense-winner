# Drift-Sense: Wafer Navigation-Error Recovery

**SEMICON India Hackathon 2026 — Track 2 (Applied Materials Challenge)**

Welcome to our project! Drift-Sense solves a major problem in semiconductor manufacturing: when inspection microscopes lose their exact place on a microchip due to tiny mechanical drifts. Our AI-powered software automatically finds the "lost" location by comparing a small, high-quality reference photo with a large, noisy search area.

We solve this for **both** Grayscale SEM (electron microscopes) and RGB Optical (light microscopes), fulfilling all base requirements and bonus criteria.

---

## 🚀 Quick Start (One-Click Runner)

We've provided automated execution scripts so judges can run the entire solution locally with zero hassle. It will automatically create a secure python environment and install all dependencies.

**For Windows:**
Double-click `run_windows.bat` in your file explorer.

**For Mac / Linux:**
Open your terminal in this directory and run:
```bash
bash run_mac_linux.sh
```

---

### Manual Docker Setup (Optional)
If you prefer containerized environments, we have also provided a `Dockerfile`:
```bash
docker build -t drift-sense .
docker run -p 8501:8501 drift-sense
```


**2. Generate the Datasets (Optional)**
We have already included 30-sample curated test sets for both SEM and Optical in the `final_submission_dataset/` and `final_submission_dataset_opt/` folders. If you want to generate new ones yourself:
```bash
# Generate 30 SEM (Grayscale) samples
python -m final_data_generation.run --num-samples 30 --output-dir ./my_sem_dataset

# Generate 30 Optical (RGB) samples
python -m final_data_generation.run_optical --num-samples 30 --output-dir ./my_optical_dataset
```

**3. Run the Evaluation (Get Metrics)**
To test our algorithm on the 30-sample datasets and see the accuracy metrics (1px-5px) and computation time:
```bash
# Evaluate the SEM (Grayscale) dataset:
python evaluate.py --manifest final_submission_dataset/test/manifest.csv --tolerance-px 5 --output-dir final_submission_dataset/results

# Evaluate the Optical (RGB) dataset:
python evaluate.py --manifest final_submission_dataset_opt/manifest.csv --tolerance-px 5 --output-dir final_submission_dataset_opt/results --optical
```

**4. View the Interactive Dashboard**
To see our beautiful visual dashboard and explore the images and metrics interactively:
```bash
streamlit run ui.py
```

---

## 📊 Performance & Explainability

Our algorithm achieves sub-pixel accuracy in under 200ms per image pair, relying on a robust, highly optimized mathematical approach (ZNCC + Multi-Scale Voting + Normalized RGB) without needing heavy deep-learning models.

- **SEM (Grayscale) Accuracy:** **83.3%** at ≤5px tolerance.
- **Optical (RGB) Accuracy:** **96.7%** at ≤5px tolerance.

### 🔍 Why does SEM sometimes fail? (Explainability)
In the remaining ~17% of extreme SEM cases, our system correctly flags a `WARNING_ALIASING_RISK`. The root cause is **perfect periodicity**: if you take a tiny 100x100 pixel reference crop from the dead center of a massive DRAM array (which repeats perfectly every 10 pixels), and drift it by 400 pixels, it is physically and mathematically impossible to know which exact block you are looking at without wider macro-context. The algorithm isn't broken; it is hitting the mathematical limit of the data provided. 

Detailed analysis of our datasets and failure modes can be found in:
- `final_submission_dataset/dataset_rationale.md`

---

## 📁 Project Structure

```text
drift-sense-winner/
├── README.md                   # You are reading this!
├── requirements.txt            # Required Python packages
├── CITATIONS.md                # Academic physics papers backing our dataset generator
├── METHODOLOGY.md              # Detailed pipeline architecture & strategy
├── PRESENTATION_SCRIPT.md      # Pitch script for judges
├── ui.py                       # The Streamlit Visual Dashboard
│
├── localize.py                 # MAIN INFERENCE SCRIPT (The core routing engine)
├── evaluate.py                 # Tests the algorithm and generates accuracy graphs
│
├── final_submission_dataset/   # The curated 30-sample SEM dataset
├── final_submission_dataset_opt/ # The curated 30-sample Optical dataset
│
├── final_data_generation/      # Upgraded physics-backed dataset generators
└── src/                        # The matching algorithms (ZNCC, Strip Anchor, Optical)
```
