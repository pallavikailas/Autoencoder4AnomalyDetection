import streamlit as st
import pandas as pd
from dataloader import create_dataloaders
from training import train_model
from anomaly_detection import detect_anomalies

st.title("Anomaly Detection App")

st.write("""
# Detecting anomalies in aircraft data
This app uses an autoencoder to detect anomalies in aircraft data and explains the detected anomalies using LIME.
""")

# File uploaders for user to upload their own data files
file_1 = st.file_uploader("Choose the first Excel file (Correct data)", type="xlsx")
file_2 = st.file_uploader("Choose the second Excel file (Wrong data)", type="xlsx")

if file_1 is not None and file_2 is not None:
    # Create DataLoaders
    train_dataloader, val_dataloader, test_dataloader = create_dataloaders(file_1, file_2)

    # Train the model
    autoencoder = train_model(train_dataloader, val_dataloader)

    # Perform anomaly detection
    anomalies_df = detect_anomalies(test_dataloader, autoencoder)

    st.write("## Anomalies Detected")
    st.dataframe(anomalies_df)

# Run the Streamlit app
if __name__ == '__main__':
    st.set_page_config(page_title="Anomaly Detection", layout="wide")
    st.experimental_rerun()
