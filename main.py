from dataloader import test_dataloader
from training import train_model
from anomaly_detection import detect_anomalies
import streamlit as st
import pandas as pd


st.title("Anomaly Detection App")

st.write("""
# Detecting anomalies in aircraft data
This app uses an autoencoder to detect anomalies in aircraft data and explains the detected anomalies using LIME.
""")

# Display the uploaded file's dataframe
anomalies_df = detect_anomalies(test_dataloader)

st.write("## Anomalies Detected")
st.dataframe(anomalies_df)

st.write("## Detailed Anomaly Explanations")
for index, row in anomalies_df.iterrows():
    st.write(f"### Anomaly {index + 1}")
    st.write(f"A/C Registration: {row['A/C Registration']}")
    st.write(f"Arr Airport: {row['Arr Airport']}")
    st.write(f"Dep Airport: {row['Dep Airport']}")
    st.write(f"A/C Weight: {row['A/C weight']}")
    st.write(f"Arr Weight: {row['Arr weight']}")
    st.write(f"Dep Weight: {row['Dep weight']}")
    st.write(f"Total Weight: {row['Total weight']}")

# Run the Streamlit app
if __name__ == '__main__':
    train_model()
    st.set_page_config(page_title="Anomaly Detection", layout="wide")
    st.experimental_rerun()
    
