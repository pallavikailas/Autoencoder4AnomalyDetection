import streamlit as st
import pandas as pd
from torch.utils.data import DataLoader
from training import train_model
from anomaly_detection import detect_anomalies
from dataloader import ExcelDataset  # Ensure this is your dataset class

st.title("Anomaly Detection App")

st.write("""
# Detecting anomalies in aircraft data
This app uses an autoencoder to detect anomalies in aircraft data and explains the detected anomalies using LIME.
""")

uploaded_file_1 = st.file_uploader("Choose the first Excel file (Correct data)", type="xlsx")
uploaded_file_2 = st.file_uploader("Choose the second Excel file (Wrong data)", type="xlsx")

if uploaded_file_1 is not None and uploaded_file_2 is not None:
    # Read the uploaded Excel files
    correct_data = pd.read_excel(uploaded_file_1)
    wrong_data = pd.read_excel(uploaded_file_2)

    # Concatenate the two datasets
    combined_data = pd.concat([correct_data, wrong_data], axis=0)

    # Assuming your dataset class can take a DataFrame directly
    dataset = ExcelDataset(combined_data)
    test_dataloader = DataLoader(dataset, batch_size=32, shuffle=False)  # Adjust as necessary

    # Train the model
    train_model()

    # Perform anomaly detection
    anomalies_df = detect_anomalies(test_dataloader)

    st.write("## Anomalies Detected")
    st.dataframe(anomalies_df)

# Run the Streamlit app
if __name__ == '__main__':
    st.set_page_config(page_title="Anomaly Detection", layout="wide")
    st.experimental_rerun()
