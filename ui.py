import streamlit as st
import subprocess
import os
import pandas as pd

st.set_page_config(page_title="Drift-Sense Dashboard", layout="wide")

st.title("🔬 Drift-Sense Visual Dashboard")
st.markdown("A control panel for the Drift-Sense SEM localization engine.")

# Sidebar for Configuration
st.sidebar.header("Configuration")

# Customization Mode
config_mode = st.sidebar.radio("Customization Mode", ["Severity Curriculum (Easy)", "Advanced Customization (Full Physics)"])

# Batch or Single
batch_mode = st.sidebar.radio("Generation Mode", ["Single Image", "Batch Mode"])
num_samples = 1 if batch_mode == "Single Image" else st.sidebar.number_input("Number of Samples", min_value=1, max_value=500, value=10)

severity = None
advanced_args = []

if config_mode == "Severity Curriculum (Easy)":
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
output_dir = st.sidebar.text_input("Dataset Directory", value="./data_streamlit")
results_dir = st.sidebar.text_input("Results Directory", value="./results_streamlit")

st.sidebar.markdown("---")
st.sidebar.markdown("### Action Controls")

col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Data Generation")
    st.write("Generate a synthetic physical SEM dataset based on the severity curriculum.")
    if st.button("Generate Dataset", use_container_width=True):
        if config_mode == "Severity Curriculum (Easy)":
            st.info(f"Generating {num_samples} samples at Severity {severity}...")
        else:
            st.info(f"Generating {num_samples} custom physics samples...")
            
        # Build command
        cmd = [
            "python3", "-m", "final_data_generation.run",
            "--num-samples", str(num_samples),
            "--output-dir", output_dir
        ]
        
        if config_mode == "Severity Curriculum (Easy)":
            cmd.extend(["--severity-level", str(severity)])
        else:
            cmd.extend(advanced_args)
        
        with st.spinner("Running final_data_generation engine..."):
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                st.success("Dataset generated successfully!")
                with st.expander("View Generation Logs", expanded=True):
                    st.code(result.stdout + "\n" + result.stderr)
            else:
                st.error("Error generating dataset.")
                st.code(result.stderr)

with col2:
    st.subheader("2. Inference & Evaluation")
    st.write("Run the routing engine (ZNCC + U-Net) and calculate sub-pixel accuracy.")
    if st.button("Run Evaluation", use_container_width=True):
        st.info("Running evaluation pipeline...")
        
        manifest_path = os.path.join(output_dir, "test", "manifest.csv")
        
        if not os.path.exists(manifest_path):
            st.error(f"Manifest file not found at {manifest_path}. Please generate the dataset first.")
        else:
            # Build command
            cmd = [
                "python3", "evaluate.py",
                "--manifest", manifest_path,
                "--tolerance-px", "5",
                "--output-dir", results_dir
            ]
            
            with st.spinner("Running inference engine (this may take a moment)..."):
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    st.success("Evaluation completed successfully!")
                    with st.expander("View Evaluation Logs", expanded=True):
                        st.code(result.stdout)
                    
                    # Display the results summary if available
                    summary_path = os.path.join(results_dir, "results_summary.txt")
                    if os.path.exists(summary_path):
                        st.markdown("### Final Results")
                        with open(summary_path, "r") as f:
                            st.text(f.read())
                else:
                    st.error("Error running evaluation.")
                    st.code(result.stderr)

st.markdown("---")
st.header("📊 Evaluation Analysis")

if os.path.exists(results_dir):
    tab1, tab2 = st.tabs(["Predictions Table", "Failure Analysis"])
    
    with tab1:
        st.subheader("Ground Truth vs Predicted Locations")
        predictions_path = os.path.join(results_dir, "predictions.csv")
        if os.path.exists(predictions_path):
            df = pd.read_csv(predictions_path)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("Run Evaluation to generate the predictions table.")
            
    with tab2:
        st.subheader("Why did the engine fail?")
        failure_txt_path = os.path.join(results_dir, "failure_analysis.txt")
        failure_img_path = os.path.join(results_dir, "failure_case.png")
        
        if os.path.exists(failure_txt_path):
            with open(failure_txt_path, "r") as f:
                failure_text = f.read()
            st.text(failure_text)
        else:
            st.info("No failure analysis found.")
            
        if os.path.exists(failure_img_path):
            st.image(failure_img_path, use_container_width=True, caption="Worst Failure Case")
else:
    st.info("Run Evaluation above to unlock the Analysis section!")

st.markdown("---")
st.header("🖼️ Dataset Image Viewer")

test_dir = os.path.join(output_dir, "test")
if os.path.exists(test_dir):
    # Find all reference images to determine available samples
    ref_dir = os.path.join(test_dir, "reference")
    if os.path.exists(ref_dir):
        images = sorted([f for f in os.listdir(ref_dir) if f.endswith(".png")])
        if images:
            selected_img = st.selectbox("Select a sample to view:", images)
            
            col_ref, col_search = st.columns(2)
            
            with col_ref:
                st.subheader("Reference Image")
                st.write("(Clean, 1nm/pixel, High Dose)")
                st.image(os.path.join(ref_dir, selected_img), use_container_width=True)
                
            with col_search:
                st.subheader("Search Image")
                st.write("(Drifted, 10nm/pixel, Low Dose)")
                search_path = os.path.join(test_dir, "search", selected_img)
                if os.path.exists(search_path):
                    st.image(search_path, use_container_width=True)
                else:
                    st.error("Search image not found.")
        else:
            st.info("No images found in the dataset folder.")
else:
    st.info("Generate a dataset above to unlock the Image Viewer!")
