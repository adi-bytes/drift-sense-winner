import streamlit as st
import subprocess
import os

st.set_page_config(page_title="Drift-Sense Dashboard", layout="wide")

st.title("🔬 Drift-Sense Visual Dashboard")
st.markdown("A control panel for the Drift-Sense SEM localization engine.")

# Sidebar for Configuration
st.sidebar.header("Configuration")
severity = st.sidebar.slider("Severity Level (0-6)", min_value=0, max_value=6, value=2, 
                            help="0=Ideal, 6=Extreme Drift & Noise")
num_samples = st.sidebar.number_input("Number of Samples", min_value=1, max_value=500, value=10)
output_dir = st.sidebar.text_input("Dataset Directory", value="./data_streamlit")
results_dir = st.sidebar.text_input("Results Directory", value="./results_streamlit")

st.sidebar.markdown("---")
st.sidebar.markdown("### Action Controls")

col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Data Generation")
    st.write("Generate a synthetic physical SEM dataset based on the severity curriculum.")
    if st.button("Generate Dataset", use_container_width=True):
        st.info(f"Generating {num_samples} samples at Severity {severity}...")
        
        # Build command
        cmd = [
            "python3", "-m", "final_data_generation.run",
            "--num-samples", str(num_samples),
            "--severity-level", str(severity),
            "--output-dir", output_dir
        ]
        
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
        
        manifest_path = os.path.join(output_dir, "manifest.csv")
        
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
