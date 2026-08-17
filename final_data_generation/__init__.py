"""
final_data_generation — upgraded synthetic SEM dataset generator.

Preserves all original DRAM/FinFET/zones/physics from src/ but adds:
  - Smooth temporal drift trajectory (GP-style random walk)
  - Correlated scan-line shifts (separate from drift)
  - Correlated electronic noise (FFT spatial filtering)
  - LER/LWR correlated edge roughness on geometry
  - Sidewall angle as grayscale gradient inside line bodies
  - Material-layer SE gain factors
  - Physics severity curriculum (--severity-level 0-6)

Entry point:
    python final_data_generation/run.py --num-samples 30 --severity-level 3
"""
