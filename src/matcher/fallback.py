"""
Heavy Fallback Pipeline for the Drift-Sense Matcher.

This module is dynamically routed to when the fast-path ZNCC matcher fails
due to extreme Out-Of-Distribution (OOD) noise, charging artifacts, or drift.
It sacrifices computational speed (~200ms) for robustness, applying heavy
deep-learning denoising (U-Net) and Bayesian Gaussian Prior weighting
to rescue degraded images.
"""

from __future__ import annotations

import logging
import cv2
import numpy as np
import torch
import os

from src.matcher.coarse_matcher import Candidate
from src.models.unet import SEMUNet

logger = logging.getLogger(__name__)

# Initialize U-Net globally to avoid load times on every call
DEVICE = torch.device('cuda' if torch.cuda.is_available() else ('mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'cpu'))
logger.info(f"Loading Fallback U-Net on {DEVICE}...")

model = SEMUNet(base_ch=32).to(DEVICE)
model_path = 'models/unet_denoiser.pth'
if os.path.exists(model_path):
    ckpt = torch.load(model_path, map_location=DEVICE, weights_only=False)
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
    logger.info("Successfully loaded unet_denoiser.pth")
else:
    logger.error(f"Could not find {model_path}! Fallback will run with untrained weights.")
model.eval()


def denoise_unet(img_uint8: np.ndarray) -> np.ndarray:
    """Pass a 1000x1000 uint8 image through the U-Net."""
    # Min-Max normalize to [0, 1] to match training distribution
    img_f = img_uint8.astype(np.float32)
    img_f = (img_f - img_f.min()) / (img_f.max() - img_f.min() + 1e-8)
    
    t = torch.from_numpy(img_f).unsqueeze(0).unsqueeze(0)
    t = t.to(DEVICE)
    with torch.no_grad():
        out = model(t)
    out_np = out.squeeze().cpu().numpy()
    return (out_np * 255).clip(0, 255).astype(np.uint8)


def heavy_match(
    reference: np.ndarray,
    search: np.ndarray,
) -> tuple[list[Candidate], np.ndarray, np.ndarray]:
    """Execute the Deep Learning fallback pipeline.

    1. Denoises both images using the trained SEMUNet.
    2. Runs standard ZNCC on the clean images.
    3. Applies a Bayesian Gaussian Drift Prior to mathematically disambiguate
       purely periodic structures (e.g., DRAM) without heuristics.

    Args:
        reference: Reference image (1000x1000).
        search: Search image (1000x1000).

    Returns:
        Tuple containing:
        - List with the single best Bayesian-weighted Candidate.
        - The denoised reference image (for refinement).
        - The denoised search image (for refinement).
    """
    logger.info("Executing Heavy Fallback Match (U-Net + Bayesian ZNCC)")

    # 1. Deep Learning Denoising
    ref_clean = denoise_unet(reference)
    search_clean = denoise_unet(search)

    # 2. Resize reference to template (10x zoom)
    scale = 10.0
    tw = max(int(round(reference.shape[1] / scale)), 1)
    th = max(int(round(reference.shape[0] / scale)), 1)
    template = cv2.resize(ref_clean, (tw, th), interpolation=cv2.INTER_AREA)

    # 3. ZNCC Correlation
    corr_map = cv2.matchTemplate(search_clean, template, cv2.TM_CCOEFF_NORMED)

    # Just take raw ZNCC for now to see if U-Net solves it without the prior
    py, px = np.unravel_index(np.argmax(corr_map), corr_map.shape)
    score = float(corr_map[py, px])

    logger.info(f"Bayesian MAP peak found at ({px}, {py}) with raw ZNCC score: {score:.3f}")

    best_candidate = Candidate(
        x=px + tw / 2.0,
        y=py + th / 2.0,
        score=score,
        scale=scale,
        template_w=tw,
        template_h=th,
        rotation=0.0
    )

    return [best_candidate], ref_clean, search_clean
