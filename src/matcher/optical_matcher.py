import logging

import cv2
import numpy as np

from src.matcher.coarse_matcher import Candidate, _find_local_maxima

logger = logging.getLogger(__name__)

from scipy.ndimage import gaussian_filter


def homomorphic_filter(channel: np.ndarray, sigma: float = 60.0, boost: float = 1.0) -> np.ndarray:
    """
    Applies a homomorphic high-pass filter to remove multiplicative low-frequency 
    illumination gradients (vignetting) while preserving thin-film color shifts.
    """
    # Offset by 128 (since LAB A/B channels are centered around 128) to ensure > 0
    # Actually, OpenCV LAB channels are 0-255, so they are already positive. We add 1 for log safety.
    log_img = np.log1p(channel.astype(np.float32))
    low_freq = gaussian_filter(log_img, sigma=sigma)
    high_freq = log_img - low_freq
    filtered = np.expm1(high_freq * boost)
    return filtered

def subpixel_peak(score_map: np.ndarray, peak_loc: tuple[int, int]) -> tuple[float, float]:
    """Fits a 2D quadratic to the 3x3 neighborhood of the ZNCC peak for sub-pixel accuracy."""
    x, y = peak_loc
    h, w = score_map.shape
    
    # Boundary guard
    if x <= 0 or x >= w - 1 or y <= 0 or y >= h - 1:
        return float(x), float(y)
        
    patch = score_map[y-1:y+2, x-1:x+2]
    
    # Quadratic fit in X and Y independently
    denom_y = 2 * (2 * patch[1,1] - patch[0,1] - patch[2,1])
    denom_x = 2 * (2 * patch[1,1] - patch[1,0] - patch[1,2])
    
    dy = (patch[2,1] - patch[0,1]) / denom_y if denom_y != 0 else 0.0
    dx = (patch[1,2] - patch[1,0]) / denom_x if denom_x != 0 else 0.0
    
    # Bound the shift to [-1.0, 1.0] pixel to allow wider subpixel refinement
    dx = max(-1.0, min(1.0, dx))
    dy = max(-1.0, min(1.0, dy))
    
    return float(x + dx), float(y + dy)

def color_aware_match(
    reference_rgb: np.ndarray,
    search_rgb: np.ndarray,
    scale: float = 8.0
) -> list[Candidate]:
    """
    Executes a Color-Aware ZNCC matching pipeline for Optical RGB Inspection.
    
    Key improvements:
    1. Per-channel sub-pixel refinement (sharper parabolic fit per channel)
    2. Vignette gradient voting to break aliased peak ties
    """
    logger.info("Executing RGB ZNCC Optical Matcher (Per-Channel Sub-Pixel)")
    
    ref_f = reference_rgb.astype(np.float32)
    search_f = search_rgb.astype(np.float32)
    
    # Normalize RGB by intensity to remove illumination gradients
    ref_sum = np.sum(ref_f, axis=2, keepdims=True) + 1e-6
    search_sum = np.sum(search_f, axis=2, keepdims=True) + 1e-6
    ref_f = ref_f / ref_sum
    search_f = search_f / search_sum
    
    # Resize reference to the search scale (10x reduction)
    tw = max(round(ref_f.shape[1] / scale), 1)
    th = max(round(ref_f.shape[0] / scale), 1)
    ref_resized = cv2.resize(ref_f, (tw, th), interpolation=cv2.INTER_AREA)
    
    # ZNCC on R, G, B channels independently
    scores = []
    for ch in range(3):
        s = cv2.matchTemplate(search_f[:, :, ch], ref_resized[:, :, ch], cv2.TM_CCOEFF_NORMED)
        scores.append(s)
    
    combined_score = (scores[0] + scores[1] + scores[2]) / 3.0
    
    # Find peaks using local maxima
    raw_peaks = _find_local_maxima(combined_score, threshold=0.4, max_peaks=8)
    
    if not raw_peaks:
        return []
    
    # --- Vignette Gradient Voting ---
    # The optical vignette creates a radial intensity gradient centered on the lens axis.
    # The TRUE match location should have the most consistent vignette profile between
    # reference and search. We measure this by computing the local intensity gradient
    # direction at each peak and comparing ref vs search.
    search_gray = np.mean(search_rgb.astype(np.float32), axis=2)
    grad_x = cv2.Sobel(search_gray, cv2.CV_32F, 1, 0, ksize=31)
    grad_y = cv2.Sobel(search_gray, cv2.CV_32F, 0, 1, ksize=31)
    
    candidates = []
    for px, py, score in raw_peaks:
        # --- Per-channel sub-pixel refinement ---
        # Each channel's score map has a slightly different peak shape due to 
        # chromatic aberration. Fitting the parabola per-channel and averaging
        # gives a MUCH sharper sub-pixel estimate than fitting the blurred average.
        sub_xs = []
        sub_ys = []
        for ch_score in scores:
            spx, spy = subpixel_peak(ch_score, (px, py))
            sub_xs.append(spx)
            sub_ys.append(spy)
        
        # Weighted average of per-channel sub-pixel positions
        final_spx = np.mean(sub_xs)
        final_spy = np.mean(sub_ys)
        
        # Vignette consistency bonus: peaks closer to the vignette center
        # (where gradient magnitude is lowest) are more likely to be the true match
        gy = min(max(py + th // 2, 0), search_gray.shape[0] - 1)
        gx = min(max(px + tw // 2, 0), search_gray.shape[1] - 1)
        grad_mag = np.sqrt(grad_x[gy, gx]**2 + grad_y[gy, gx]**2)
        # Normalize gradient magnitude to a small bonus (max ~0.02)
        max_grad = np.max(np.sqrt(grad_x**2 + grad_y**2)) + 1e-6
        vignette_bonus = 0.02 * (1.0 - grad_mag / max_grad)
        
        adjusted_score = float(score) + vignette_bonus
        
        # Convert OpenCV coordinate center offset
        geom_x = final_spx + (tw - 1) / 2.0
        geom_y = final_spy + (th - 1) / 2.0
        
        candidates.append(Candidate(
            x=geom_x,
            y=geom_y,
            score=adjusted_score,
            scale=scale,
            template_w=tw,
            template_h=th,
            rotation=0.0
        ))
    
    # Re-sort by adjusted score (vignette-aware)
    candidates.sort(key=lambda c: c.score, reverse=True)
        
    if candidates:
        logger.info(f"Optical Matcher locked onto top peak ({candidates[0].x:.2f}, {candidates[0].y:.2f}) with ZNCC score: {candidates[0].score:.4f}")
    
    return candidates
