import os
import subprocess

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Drift-Sense Dashboard", layout="wide")

st.title("🔬 Drift-Sense Visual Dashboard")
st.markdown("A control panel for the Drift-Sense SEM localization engine.")

# Sidebar for Configuration
st.sidebar.header("Configuration")

# Track Mode
track_mode = st.sidebar.radio("Target Modality", ["Grayscale (SEM)", "RGB (Optical)"])

# Customization Mode
config_mode = st.sidebar.radio("Customization Mode", ["Severity Curriculum (Levels)", "Advanced Customization (Full Physics)"])

# Batch or Single
batch_mode = st.sidebar.radio("Generation Mode", ["Single Image", "Batch Mode"])

if batch_mode == "Single Image":
    num_dram = 1
    num_finfet = 0
else:
    num_dram = st.sidebar.number_input("Number of DRAM Samples", min_value=0, max_value=250, value=5)
    num_finfet = st.sidebar.number_input("Number of FinFET Samples", min_value=0, max_value=250, value=5)

severity = None
advanced_args = []

if config_mode == "Severity Curriculum (Levels)":
    severity = st.sidebar.slider("Severity Level (0-6)", min_value=0, max_value=6, value=2, 
                                help="0=Ideal, 6=Extreme Drift & Noise")
else:
    st.sidebar.markdown("### Advanced Physics Knobs")
    dose_ref = st.sidebar.slider("Reference Dose (higher = cleaner)", 500, 5000, 2000)
    dose_search = st.sidebar.slider("Search Dose (higher = cleaner)", 50, 2000, 200)
    stage_drift = st.sidebar.slider("Stage Placement Error (px)", 0, 450, 50,
                                   help="Simulates macro-stage navigation errors")
    shear_amp = st.sidebar.slider("Temporal Drift Amplitude (px)", 0.0, 10.0, 1.5,
                                 help="Smooth raster drift shear")
    noise_sigma = st.sidebar.slider("Electronic Noise Sigma", 0.0, 25.0, 5.0)
    spot_size = st.sidebar.slider("Beam Spot Size (nm)", 1.0, 20.0, 5.0)
    advanced_args = [
        "--dose-reference", str(dose_ref),
        "--dose-search", str(dose_search),
        "--stage-drift-px", str(stage_drift),
        "--drift-amplitude-px", str(shear_amp),
        "--correlated-noise-sigma", str(noise_sigma),
        "--beam-spot-size-nm", str(spot_size),
    ]

st.sidebar.markdown("---")
default_out = "./data_streamlit_opt" if "RGB" in track_mode else "./data_streamlit_sem"
default_res = "./results_streamlit_opt" if "RGB" in track_mode else "./results_streamlit_sem"
output_dir = st.sidebar.text_input("Dataset Directory", value=default_out)
results_dir = st.sidebar.text_input("Results Directory", value=default_res)

st.sidebar.markdown("---")
st.sidebar.markdown("### Action Controls")

tab1, tab2, tab3 = st.tabs(["🔬 1. Data Gen & Viewer", "⚙️ 2. Run Inference", "📊 3. Dashboard & Analysis"])

# --- TAB 1: DATA GENERATION & VIEWER ---
with tab1:
    st.subheader("Data Generation Engine")
    st.write("Generate a synthetic physical SEM dataset based on your sidebar curriculum.")
    
    col_btn, col_msg = st.columns([1, 2])
    with col_btn:
        generate_clicked = st.button("Generate Dataset", width="stretch", type="primary")
        
    if generate_clicked:
        total_samples = num_dram + num_finfet
        if config_mode == "Severity Curriculum (Levels)":
            st.info(f"Generating {total_samples} samples at Severity {severity}...")
        else:
            st.info(f"Generating {total_samples} custom physics samples...")
            
        script = "final_data_generation.run_optical" if "RGB" in track_mode else "final_data_generation.run"
        cmd = [
            "python3", "-m", script,
            "--num-dram", str(num_dram),
            "--num-finfet", str(num_finfet),
            "--output-dir", output_dir
        ]
        
        if "SEM" in track_mode:
            if config_mode == "Severity Curriculum (Levels)":
                cmd.extend(["--severity-level", str(severity)])
            else:
                cmd.extend(advanced_args)
        else:
            st.warning("Optical generator uses fixed physics presets. Ignoring curriculum/advanced sliders.")
        
        with st.spinner("Running final_data_generation engine..."):
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                st.success(f"Successfully generated {total_samples} samples!")
                with st.expander("View Generation Logs", expanded=False):
                    st.code(result.stdout + "\n" + result.stderr)
            else:
                st.error("Error generating dataset.")
                with st.expander("View Error Logs", expanded=True):
                    st.code(result.stderr)

    st.markdown("---")
    st.subheader("Dataset Image Viewer")
    
    test_dir = output_dir if "RGB" in track_mode else os.path.join(output_dir, "test")
    if os.path.exists(test_dir):
        ref_dir = os.path.join(test_dir, "reference")
        if os.path.exists(ref_dir):
            images = sorted([f for f in os.listdir(ref_dir) if f.endswith(".png")])
            if images:
                selected_img = st.selectbox("Select a generated sample to view:", images)
                
                col_ref, col_search = st.columns(2)
                with col_ref:
                    st.markdown("**Reference Image** *(Clean, 1nm/pixel, High Dose)*")
                    st.image(os.path.join(ref_dir, selected_img), width="stretch")
                with col_search:
                    st.markdown("**Search Image** *(Drifted, 10nm/pixel, Low Dose)*")
                    search_path = os.path.join(test_dir, "search", selected_img)
                    if os.path.exists(search_path):
                        st.image(search_path, width="stretch")
                    else:
                        st.error("Search image not found.")
            else:
                st.info("No images found in the dataset folder.")
    else:
        st.info("Click 'Generate Dataset' above to populate the viewer.")


# --- TAB 2: RUN INFERENCE ---
with tab2:
    st.subheader("Localization & Evaluation Engine")
    st.write("Run the Adaptive Cascade routing engine (ZNCC + Strip Anchors) and calculate sub-pixel accuracy.")
    
    col_run, col_run_msg = st.columns([1, 2])
    with col_run:
        eval_clicked = st.button("Run Evaluation Pipeline", width="stretch", type="primary")
        
    if eval_clicked:
        manifest_path = os.path.join(output_dir, "manifest.csv") if "RGB" in track_mode else os.path.join(output_dir, "test", "manifest.csv")
        if not os.path.exists(manifest_path):
            st.error(f"Manifest file not found at {manifest_path}. Please generate the dataset first.")
        else:
            tol_px = "15" if "RGB" in track_mode else "5"
            cmd = [
                "python3", "evaluate.py",
                "--manifest", manifest_path,
                "--tolerance-px", tol_px,
                "--output-dir", results_dir
            ]
            if "RGB" in track_mode:
                cmd.append("--optical")
            
            with st.spinner("Running inference engine (this may take a moment)..."):
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    st.success("Evaluation completed successfully!")
                    with st.expander("View Raw Evaluation Logs", expanded=False):
                        st.code(result.stdout)
                else:
                    st.error("Error running evaluation.")
                    with st.expander("View Error Logs", expanded=True):
                        st.code(result.stderr)
    
    st.info("Once evaluation is complete, proceed to the Dashboard tab to view the visual breakdown.")


# --- TAB 3: DASHBOARD & ANALYSIS ---
with tab3:
    if not os.path.exists(results_dir) or not os.path.exists(os.path.join(results_dir, "predictions.csv")):
        st.warning("No evaluation results found. Please run the evaluation pipeline in Tab 2 first.")
    else:
        # High Level Metrics
        predictions_path = os.path.join(results_dir, "predictions.csv")
        df = pd.read_csv(predictions_path)
        
        thresholds = [1, 2, 3, 4, 5, 6, 9, 12, 15] if "RGB" in track_mode else [1, 2, 3, 4, 5]
        
        accs = [(df["error"] <= t).mean() * 100 for t in thresholds]
        mean_err = df["error"].mean()
        median_err = df["error"].median()
        mean_time = df["time"].mean() * 1000  # ms
        
        st.subheader("Performance Metrics")
        
        st.markdown(f"**Accuracies ({thresholds[0]}px - {thresholds[-1]}px Tolerance)**")
        
        # Display metrics in chunks of 5 columns
        for i in range(0, len(thresholds), 5):
            cols = st.columns(5)
            chunk = thresholds[i:i+5]
            for j, t in enumerate(chunk):
                cols[j].metric(f"{t}px", f"{accs[i+j]:.1f}%")

        with st.expander("View Confusion Matrix (Matches vs Mismatches)"):
            cm_data = []
            for tol in thresholds:
                matches = (df["error"] <= tol).sum()
                mismatches = len(df) - matches
                acc = (matches / len(df)) * 100
                cm_data.append({"Tolerance": f"<= {tol}px", "Match (TP)": matches, "Mismatch (FP)": mismatches, "Accuracy": f"{acc:.1f}%"})
            st.table(pd.DataFrame(cm_data).set_index("Tolerance"))
        
        st.markdown("**Error & Latency**")
        m1, m2, m3 = st.columns(3)
        m1.metric("Mean Error", f"{mean_err:.2f} px")
        m2.metric("Median Error", f"{median_err:.2f} px")
        m3.metric("Avg Latency", f"{mean_time:.0f} ms")
        
        st.markdown("---")
        
        # Explainability & Failure Analysis
        st.subheader("Explainability & Root Cause Analysis")
        failure_txt_path = os.path.join(results_dir, "failure_analysis.txt")
        failure_img_path = os.path.join(results_dir, "failure_case.png")
        
        if os.path.exists(failure_txt_path) and os.path.exists(failure_img_path):
            col_img, col_txt = st.columns([3, 2])
            
            with col_img:
                st.markdown("**Worst Failure Case Visualization**")
                st.image(failure_img_path, width="stretch")
                
            with col_txt:
                st.markdown("**Root Cause Diagnosis**")
                with open(failure_txt_path, "r") as f:
                    failure_text = f.read()
                
                # Parse the text file to make it look nicer
                if "ROOT CAUSE ANALYSIS" in failure_text:
                    parts = failure_text.split("ROOT CAUSE ANALYSIS")
                    header = parts[0]
                    body = parts[1]
                    
                    if "PROPOSED FIX" in body:
                        subparts = body.split("PROPOSED FIX")
                        cause = subparts[0].replace("-", "").strip()
                        fix = subparts[1].replace("-", "").strip()
                        
                        st.error(f"**The Problem:**\n\n{cause}")
                        st.success(f"**Proposed Fix:**\n\n{fix}")
                    else:
                        st.info(body.replace("-", "").strip())
                else:
                    st.text(failure_text)
        else:
            st.info("No failure analysis data generated.")
            
        st.markdown("---")
        
        # Predictions Table
        st.subheader("Detailed Predictions Table")
        
        # Style the dataframe: green if error <= 5, red if > 5
        def highlight_errors(val):
            color = '#2ecc71' if val <= 5.0 else '#e74c3c'
            return f'color: {color}; font-weight: bold'
            
        st.dataframe(
            df.style.map(highlight_errors, subset=['error']), 
            width="stretch",
            height=300
        )
