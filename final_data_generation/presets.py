"""
Architecture presets (nm) + Physics Severity Curriculum.

Severity levels bundle correlated parameter sets so the user
doesn't need to set 25 individual flags. Each level is a superset
of the previous in terms of acquisition difficulty.

REF: Villarrubia et al. SPIE 2003/4 — noise/beam/dose relationships.
REF: Maraghechi et al. Ultramicroscopy 2018 — drift/scan-line shift magnitudes.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Architecture Presets (identical to src/presets.py — preserved verbatim)
# ---------------------------------------------------------------------------

DRAM_1X: dict = {
    "kind": "dram",
    "feature_size_nm": 32,
    "word_line_pitch_nm": 64,
    "word_line_width_nm": 32,
    "bit_line_pitch_nm": 96,
    "bit_line_width_nm": 32,
    "contact_diameter_nm": 32,
}
DRAM_DENSE: dict = {
    "kind": "dram",
    "feature_size_nm": 24,
    "word_line_pitch_nm": 48,
    "word_line_width_nm": 24,
    "bit_line_pitch_nm": 72,
    "bit_line_width_nm": 24,
    "contact_diameter_nm": 24,
}
DRAM_LOOSE: dict = {
    "kind": "dram",
    "feature_size_nm": 48,
    "word_line_pitch_nm": 96,
    "word_line_width_nm": 48,
    "bit_line_pitch_nm": 144,
    "bit_line_width_nm": 48,
    "contact_diameter_nm": 48,
}
DRAM_WIDE: dict = {
    "kind": "dram",
    "feature_size_nm": 60,
    "word_line_pitch_nm": 120,
    "word_line_width_nm": 56,
    "bit_line_pitch_nm": 180,
    "bit_line_width_nm": 60,
    "contact_diameter_nm": 58,
}
DRAM_COMPACT: dict = {
    "kind": "dram",
    "feature_size_nm": 36,
    "word_line_pitch_nm": 72,
    "word_line_width_nm": 30,
    "bit_line_pitch_nm": 108,
    "bit_line_width_nm": 34,
    "contact_diameter_nm": 30,
}
DRAM_LEGACY: dict = {
    "kind": "dram",
    "feature_size_nm": 80,
    "word_line_pitch_nm": 160,
    "word_line_width_nm": 78,
    "bit_line_pitch_nm": 240,
    "bit_line_width_nm": 80,
    "contact_diameter_nm": 78,
}
FINFET_10NM: dict = {
    "kind": "finfet",
    "fin_pitch_nm": 48,
    "fin_width_nm": 16,
    "gate_pitch_nm": 90,
    "gate_length_nm": 28,
    "contact_size_nm": 28,
}
FINFET_7NM: dict = {
    "kind": "finfet",
    "fin_pitch_nm": 40,
    "fin_width_nm": 14,
    "gate_pitch_nm": 76,
    "gate_length_nm": 24,
    "contact_size_nm": 24,
}
FINFET_14NM: dict = {
    "kind": "finfet",
    "fin_pitch_nm": 60,
    "fin_width_nm": 20,
    "gate_pitch_nm": 110,
    "gate_length_nm": 34,
    "contact_size_nm": 34,
}
FINFET_22NM: dict = {
    "kind": "finfet",
    "fin_pitch_nm": 80,
    "fin_width_nm": 26,
    "gate_pitch_nm": 150,
    "gate_length_nm": 46,
    "contact_size_nm": 44,
}
FINFET_28NM: dict = {
    "kind": "finfet",
    "fin_pitch_nm": 96,
    "fin_width_nm": 32,
    "gate_pitch_nm": 180,
    "gate_length_nm": 56,
    "contact_size_nm": 52,
}
FINFET_45NM: dict = {
    "kind": "finfet",
    "fin_pitch_nm": 140,
    "fin_width_nm": 46,
    "gate_pitch_nm": 260,
    "gate_length_nm": 80,
    "contact_size_nm": 76,
}

PRESETS: dict[str, dict] = {
    "dram_1x": DRAM_1X,
    "dram_dense": DRAM_DENSE,
    "dram_loose": DRAM_LOOSE,
    "dram_wide": DRAM_WIDE,
    "dram_compact": DRAM_COMPACT,
    "dram_legacy": DRAM_LEGACY,
    "finfet_10nm": FINFET_10NM,
    "finfet_7nm": FINFET_7NM,
    "finfet_14nm": FINFET_14NM,
    "finfet_22nm": FINFET_22NM,
    "finfet_28nm": FINFET_28NM,
    "finfet_45nm": FINFET_45NM,
}
DRAM_PRESET_NAMES = ["dram_1x", "dram_dense", "dram_loose", "dram_wide", "dram_compact", "dram_legacy"]
FINFET_PRESET_NAMES = ["finfet_10nm", "finfet_7nm", "finfet_14nm", "finfet_22nm", "finfet_28nm", "finfet_45nm"]


def get_preset(name: str) -> dict:
    if name not in PRESETS:
        raise ValueError(f"Unknown preset '{name}'. Available: {list(PRESETS)}")
    return dict(PRESETS[name])


def presets_for_kind(kind: str) -> list[dict]:
    names = DRAM_PRESET_NAMES if kind == "dram" else FINFET_PRESET_NAMES
    return [get_preset(n) for n in names]


# ---------------------------------------------------------------------------
# Material SE Gain Factors
# REF: Li et al., Scanning 35 (2013) — material-dependent SE yield in CD-SEM.
# These are reduced-order gain multipliers, NOT full Monte-Carlo transport.
# Si background=1.0, poly-Si word lines slightly brighter, metal contacts
# brightest (W has highest SE yield), SiO2 trench darker.
# ---------------------------------------------------------------------------

MATERIAL_SE_GAINS: dict[str, float] = {
    "substrate":  1.00,   # Si substrate
    "word_line":  1.15,   # poly-Si, slightly higher SE yield
    "bit_line":   1.30,   # metal line (TiN/W), higher SE yield
    "contact":    1.55,   # W contact plug — highest yield
    "fin":        1.10,   # Si fin, lightly higher than substrate
    "gate":       1.25,   # poly/metal gate
    "oxide":      0.78,   # SiO2 trench — lower SE yield, darker
}


# ---------------------------------------------------------------------------
# Physics Severity Curriculum
# REF: Maraghechi et al. Ultramicroscopy 2018 — realistic drift/shift magnitudes.
# REF: Villarrubia et al. SPIE 5038 (2003) — dose/noise relationships in CD-SEM.
#
# Levels define correlated parameter bundles. Each level is a superset of
# the previous in physical difficulty. Mirrors the training curriculum concept
# from deep learning literature — start easy, progressively increase difficulty.
# ---------------------------------------------------------------------------

SEVERITY_PARAMS: dict[int, dict] = {
    0: dict(
        # Level 0: Near-ideal geometry. Minimal noise. For sanity checks.
        dose_reference=5000.0, dose_search=3000.0,
        drift_amplitude_px=0.0, drift_correlation_rows=50,
        scanline_shift_sigma_px=0.0, scanline_shift_correlation=5,
        correlated_noise_sigma=1.0, correlated_noise_length_px=1.0,
        ler_sigma_nm=0.0, ler_correlation_nm=20.0,
        sidewall_angle_deg=90.0,
        shear_amplitude_px=0.0, beam_spot_size_nm=3.0,
        astigmatism_ratio=1.0, vignette_strength=0.0,
        barrel_distortion_k=0.0, charging_streak_prob=0.0,
        speckle_sigma=0.0, salt_pepper_prob=0.0,
        rotation_max_deg=0.0, boundary_bias=0.35,
    ),
    1: dict(
        # Level 1: Normal SEM. Realistic PSF + shot noise. Fast path should win.
        dose_reference=2000.0, dose_search=800.0,
        drift_amplitude_px=0.5, drift_correlation_rows=80,
        scanline_shift_sigma_px=0.3, scanline_shift_correlation=8,
        correlated_noise_sigma=3.0, correlated_noise_length_px=2.0,
        ler_sigma_nm=1.5, ler_correlation_nm=30.0,
        sidewall_angle_deg=85.0,
        shear_amplitude_px=1.0, beam_spot_size_nm=5.0,
        astigmatism_ratio=1.05, vignette_strength=0.05,
        barrel_distortion_k=0.01, charging_streak_prob=0.01,
        speckle_sigma=0.02, salt_pepper_prob=0.001,
        rotation_max_deg=0.1, boundary_bias=0.35,
    ),
    2: dict(
        # Level 2: Low dose. Strong Poisson + correlated electronic noise.
        # Fast ZNCC starts struggling, fallback begins to activate.
        dose_reference=1500.0, dose_search=500.0,
        drift_amplitude_px=1.0, drift_correlation_rows=60,
        scanline_shift_sigma_px=0.6, scanline_shift_correlation=6,
        correlated_noise_sigma=6.0, correlated_noise_length_px=3.0,
        ler_sigma_nm=2.0, ler_correlation_nm=25.0,
        sidewall_angle_deg=82.0,
        shear_amplitude_px=1.5, beam_spot_size_nm=6.0,
        astigmatism_ratio=1.1, vignette_strength=0.1,
        barrel_distortion_k=0.02, charging_streak_prob=0.05,
        speckle_sigma=0.03, salt_pepper_prob=0.002,
        rotation_max_deg=0.2, boundary_bias=0.5,
    ),
    3: dict(
        # Level 3: Low dose + smooth temporal drift + scan-line shifts.
        # U-Net fallback required for most samples.
        dose_reference=1200.0, dose_search=400.0,
        drift_amplitude_px=2.0, drift_correlation_rows=40,
        scanline_shift_sigma_px=1.0, scanline_shift_correlation=5,
        correlated_noise_sigma=9.0, correlated_noise_length_px=4.0,
        ler_sigma_nm=2.5, ler_correlation_nm=20.0,
        sidewall_angle_deg=80.0,
        shear_amplitude_px=2.0, beam_spot_size_nm=7.0,
        astigmatism_ratio=1.15, vignette_strength=0.15,
        barrel_distortion_k=0.03, charging_streak_prob=0.1,
        speckle_sigma=0.04, salt_pepper_prob=0.003,
        rotation_max_deg=0.3, boundary_bias=0.7,
    ),
    4: dict(
        # Level 4: Lvl3 + defocus + stronger LER + proximity effects active.
        dose_reference=1000.0, dose_search=300.0,
        drift_amplitude_px=3.0, drift_correlation_rows=30,
        scanline_shift_sigma_px=1.5, scanline_shift_correlation=4,
        correlated_noise_sigma=12.0, correlated_noise_length_px=5.0,
        ler_sigma_nm=3.0, ler_correlation_nm=18.0,
        sidewall_angle_deg=78.0,
        shear_amplitude_px=2.5, beam_spot_size_nm=9.0,
        astigmatism_ratio=1.2, vignette_strength=0.2,
        barrel_distortion_k=0.04, charging_streak_prob=0.15,
        speckle_sigma=0.05, salt_pepper_prob=0.004,
        rotation_max_deg=0.4, boundary_bias=0.8,
    ),
    5: dict(
        # Level 5: Challenging SEM. High drift, noisy, strong structural variation.
        dose_reference=800.0, dose_search=200.0,
        drift_amplitude_px=4.0, drift_correlation_rows=25,
        scanline_shift_sigma_px=2.0, scanline_shift_correlation=3,
        correlated_noise_sigma=16.0, correlated_noise_length_px=6.0,
        ler_sigma_nm=3.5, ler_correlation_nm=15.0,
        sidewall_angle_deg=75.0,
        shear_amplitude_px=3.0, beam_spot_size_nm=11.0,
        astigmatism_ratio=1.3, vignette_strength=0.25,
        barrel_distortion_k=0.05, charging_streak_prob=0.2,
        speckle_sigma=0.06, salt_pepper_prob=0.005,
        rotation_max_deg=0.5, boundary_bias=0.9,
    ),
    6: dict(
        # Level 6: Extreme Drift-Sense scenario.
        # Mimics real worst-case low-dose SEM with severe stage drift.
        # U-Net fallback is the primary rescue path.
        dose_reference=600.0, dose_search=150.0,
        drift_amplitude_px=5.0, drift_correlation_rows=20,
        scanline_shift_sigma_px=2.5, scanline_shift_correlation=3,
        correlated_noise_sigma=20.0, correlated_noise_length_px=8.0,
        ler_sigma_nm=4.0, ler_correlation_nm=12.0,
        sidewall_angle_deg=72.0,
        shear_amplitude_px=4.0, beam_spot_size_nm=13.0,
        astigmatism_ratio=1.4, vignette_strength=0.3,
        barrel_distortion_k=0.06, charging_streak_prob=0.25,
        speckle_sigma=0.08, salt_pepper_prob=0.006,
        rotation_max_deg=0.6, boundary_bias=1.0,
    ),
}


def get_severity_params(level: int) -> dict:
    """Return the correlated parameter bundle for a given severity level (0-6)."""
    if level not in SEVERITY_PARAMS:
        raise ValueError(f"Severity level must be 0-6. Got: {level}")
    return dict(SEVERITY_PARAMS[level])
