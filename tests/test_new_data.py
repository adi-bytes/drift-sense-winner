import os
import cv2
import numpy as np
import pytest

import new_data

def test_generation_params_defaults():
    params = new_data.GenerationParams()
    assert params.dram_pitch_min_nm == 28.0
    assert params.rotation_max_deg == 2.5

def test_rotate_point():
    center = (0.0, 0.0)
    pt = (1.0, 0.0)
    
    # Rotate 90 degrees
    rx, ry = new_data.rotate_point(pt, center, 90.0)
    assert np.isclose(rx, 0.0, atol=1e-5)
    assert np.isclose(ry, 1.0, atol=1e-5)
    
    # Rotate -90 degrees
    rx, ry = new_data.rotate_point(pt, center, -90.0)
    assert np.isclose(rx, 0.0, atol=1e-5)
    assert np.isclose(ry, -1.0, atol=1e-5)

def test_generate_sample():
    rng = np.random.default_rng(42)
    params = new_data.GenerationParams(
        rotation_max_deg=0.0,
        barrel_distortion_k=0.0,
        blur_sigma_reference_px=0.0,
        blur_sigma_search_px=0.0,
        dose_reference=10000.0,
        dose_search=10000.0,
        charging_streak_prob=0.0,
    )
    
    sample = new_data.generate_sample("dram", rng, params)
    
    assert "reference_img" in sample
    assert "search_img" in sample
    assert "gt_x" in sample
    assert "gt_y" in sample
    
    assert sample["reference_img"].shape == (1000, 1000)
    assert sample["search_img"].shape == (1000, 1000)
    
    # Ground truth bounds
    assert 0 <= sample["gt_x"] <= 1000
    assert 0 <= sample["gt_y"] <= 1000

def test_generate_master_dram():
    rng = np.random.default_rng(42)
    params = new_data.GenerationParams()
    master, meta = new_data.generate_master("dram", 1000, rng, params)
    
    assert master.shape == (1000, 1000)
    assert "pitch_x_nm" in meta
    assert "pitch_y_nm" in meta
    assert np.max(master) > 100
    
def test_generate_master_finfet():
    rng = np.random.default_rng(42)
    params = new_data.GenerationParams()
    master, meta = new_data.generate_master("finfet", 1000, rng, params)
    
    assert master.shape == (1000, 1000)
    assert "fin_thickness_px" in meta
    assert "gate_thickness_px" in meta
    assert np.max(master) > 100
