import subprocess
import os
import shutil
import pandas as pd

out_dir = "final_submission_dataset"
os.makedirs(out_dir, exist_ok=True)

# Generate into temp folders
subprocess.run(["python3", "-m", "final_data_generation.run", "--num-samples", "15", "--severity-level", "2", "--output-dir", "tmp_lvl2", "--seed", "1"], check=True)
subprocess.run(["python3", "-m", "final_data_generation.run", "--num-samples", "10", "--severity-level", "4", "--output-dir", "tmp_lvl4", "--seed", "2"], check=True)
subprocess.run(["python3", "-m", "final_data_generation.run", "--num-samples", "5", "--severity-level", "6", "--output-dir", "tmp_lvl6", "--seed", "3"], check=True)

# Combine
test_dir = os.path.join(out_dir, "test")
os.makedirs(os.path.join(test_dir, "reference"), exist_ok=True)
os.makedirs(os.path.join(test_dir, "search"), exist_ok=True)

all_dfs = []
counter = 0
for tmp_dir in ["tmp_lvl2", "tmp_lvl4", "tmp_lvl6"]:
    df = pd.read_csv(os.path.join(tmp_dir, "test", "manifest.csv"))
    
    for idx, row in df.iterrows():
        old_id = f"{row['id']:05d}"
        new_id = f"{counter:05d}"
        
        # Move images
        shutil.copy(os.path.join(tmp_dir, "test", "reference", f"{old_id}.png"), os.path.join(test_dir, "reference", f"{new_id}.png"))
        shutil.copy(os.path.join(tmp_dir, "test", "search", f"{old_id}.png"), os.path.join(test_dir, "search", f"{new_id}.png"))
        
        # Update df
        df.at[idx, 'id'] = counter
        df.at[idx, 'reference_path'] = f"reference/{new_id}.png"
        df.at[idx, 'search_path'] = f"search/{new_id}.png"
        
        counter += 1
        
    all_dfs.append(df)

final_df = pd.concat(all_dfs, ignore_index=True)
final_df.to_csv(os.path.join(test_dir, "manifest.csv"), index=False)

# Clean up
shutil.rmtree("tmp_lvl2")
shutil.rmtree("tmp_lvl4")
shutil.rmtree("tmp_lvl6")

print(f"Successfully created curated dataset with {len(final_df)} samples!")
