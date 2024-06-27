import streamlit as st
from dataloader import create_dataloaders
from training import train_model
from anomaly_detection import detect_anomalies
from data_creater import create_dataset

def main():
    st.title("Anomaly Detection App")
    
    # File upload and data processing
    st.header("Upload Files")
    file_1 = st.file_uploader("Choose the first Excel file (Correct data)", type="xlsx")
    file_2 = st.file_uploader("Choose the second Excel file (Wrong data)", type="xlsx")
    
    if file_1 is not None and file_2 is not None:
        
        dataset = create_dataset(file_1, file_2)
        # Create DataLoaders
        train_dataloader, val_dataloader, test_dataloader = create_dataloaders(dataset)
                
        # Train autoencoder
        st.header("Training Autoencoder")
        st.write("Training in progress...")
        autoencoder = train_model(train_dataloader, val_dataloader)
        st.write("Training complete!")
        
        # Detect anomalies
        st.header("Detecting Anomalies")
        # anomalies_df = 
        detect_anomalies(test_dataloader, autoencoder, dataset)
        
        # Display anomalies
        #st.subheader("Anomalies Detected")
        #st.dataframe(anomalies_df)

if __name__ == "__main__":
    main()
