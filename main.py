from dataloader import test_dataloader
from training import train_model
from anomaly_detection import detect_anomalies

if __name__ == "__main__":
    train_model()
    detect_anomalies(test_dataloader)
