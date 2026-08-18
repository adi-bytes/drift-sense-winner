import argparse
import csv
import os

import cv2
import numpy as np

from final_data_generation.geometry import generate_zone_canvas
from final_data_generation.optical_physics import simulate_rgb_wafer_image


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--num-samples", type=int, default=10)
    p.add_argument("--output-dir", default="./optical_submission_dataset")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    
    os.makedirs(args.output_dir, exist_ok=True)
    ref_dir = os.path.join(args.output_dir, "reference")
    search_dir = os.path.join(args.output_dir, "search")
    os.makedirs(ref_dir, exist_ok=True)
    os.makedirs(search_dir, exist_ok=True)
    
    manifest_path = os.path.join(args.output_dir, "manifest.csv")
    fieldnames = ["id", "reference_path", "search_path", "gt_x", "gt_y", "architecture", "sample_seed"]
    
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for i in range(args.num_samples):
            sample_seed = int(rng.integers(0, 1_000_000))
            srng = np.random.default_rng(sample_seed)
            
            # Select random architecture
            archs = ["dram", "finfet"]
            arch = archs[srng.integers(0, len(archs))]
            
            # 1. Generate 16000x16000 (1nm/px) geometry (heightmap in nm)
            # Default height is 0 (silicon), trenches are 50nm
            zone_res = generate_zone_canvas(
                size_px=16000,
                kind=arch,
                mat_size_nm=2600.0,
                strip_width_nm=320.0,
                collapse_threshold_nm=10.0,
                linewidth_bias_nm=0.0,
                corner_rounding_px=0.0,
                rng=srng
            )
            fine_canvas = zone_res["canvas"].astype(np.float32) * 50.0 # scale to 50nm depth
            
            # 2. Pick center
            ref_cx, ref_cy = 8000, 8000
            
            # Create a correlated thickness map over the entire wafer
            thickness_map = np.full((16000, 16000), 20.0, dtype=np.float32)
            noise_lowfreq = srng.normal(0, 3.0, size=(160, 160)).astype(np.float32)
            thickness_variation = cv2.resize(noise_lowfreq, (16000, 16000), interpolation=cv2.INTER_CUBIC)
            thickness_map += thickness_variation
            
            # 3. Crop Reference (3000x3000) for physically accurate blur (context padding)
            # The Airy blur is ~440px. A 1000x1000 crop would suffer from edge reflection artifacts.
            # We simulate 3000x3000 and then crop the true 1000x1000 center to perfectly match physical optics.
            ref_crop_large = fine_canvas[ref_cy - 1500:ref_cy + 1500, ref_cx - 1500:ref_cx + 1500]
            ref_thickness_large = thickness_map[ref_cy - 1500:ref_cy + 1500, ref_cx - 1500:ref_cx + 1500]
            
            ref_img_large = simulate_rgb_wafer_image(ref_crop_large, ref_thickness_large, pixel_size_nm=1.0, photon_flux=50000, seed=sample_seed)
            
            # Crop exactly the center 1000x1000 (removing the 1000px boundary padding on all sides)
            ref_img = ref_img_large[1000:2000, 1000:2000]
            
            # 4. Crop Search Fine Patch (10000x10000), simulating stage error
            stage_err_x = int(srng.uniform(-200, 200)) * 10
            stage_err_y = int(srng.uniform(-200, 200)) * 10
            sx0 = int(ref_cx - 5000 + stage_err_x)
            sy0 = int(ref_cy - 5000 + stage_err_y)
            search_fine = fine_canvas[sy0:sy0+10000, sx0:sx0+10000]
            
            # PERFORMANCE OPTIMIZATION: Downsample topography FIRST before physics simulation
            search_fine_10nm = cv2.resize(search_fine, (1000, 1000), interpolation=cv2.INTER_AREA)
            thickness_slice = thickness_map[sy0:sy0+10000, sx0:sx0+10000]
            thickness_10nm = cv2.resize(thickness_slice, (1000, 1000), interpolation=cv2.INTER_AREA)
            
            search_img = simulate_rgb_wafer_image(search_fine_10nm, thickness_10nm, pixel_size_nm=10.0, defocus_nm=40.0, photon_flux=8000, seed=sample_seed)
            
            # Ground truth in search space
            gt_cx_search = (ref_cx - sx0) / 10.0
            gt_cy_search = (ref_cy - sy0) / 10.0
            
            ref_path = f"{i:05d}.png"
            search_path = f"{i:05d}.png"
            cv2.imwrite(os.path.join(ref_dir, ref_path), ref_img)
            cv2.imwrite(os.path.join(search_dir, search_path), search_img)
            
            writer.writerow({
                "id": i,
                "reference_path": os.path.join("reference", ref_path),
                "search_path": os.path.join("search", search_path),
                "gt_x": gt_cx_search,
                "gt_y": gt_cy_search,
                "architecture": arch,
                "sample_seed": sample_seed
            })
            print(f"Generated {i+1}/{args.num_samples}: {arch}")

if __name__ == "__main__":
    main()
