import pandas as pd
import torch
import numpy as np
from lime.lime_tabular import LimeTabularExplainer
from training import autoencoder, multiplier  
from dataloader import dataset 


def detect_anomalies(test_dataloader):
    anomalies_df = pd.DataFrame(columns=['A/C Registration', 'Arr Airport', 'Dep Airport'])
    explanations = []

    autoencoder.eval()

    feature_names = dataset.columns  

    # Convert dataset.data to numpy array if necessary
    if isinstance(dataset.data, pd.DataFrame):
        data_array = dataset.data.values
    else:
        data_array = dataset.data  # Assuming dataset.data is already a numpy array or convertible

    explainer = LimeTabularExplainer(data_array, mode='regression', feature_names=feature_names)

    with torch.no_grad():
        for inputs in test_dataloader:
            outputs, mu, logvar = autoencoder(inputs)
            recon_loss = torch.mean((inputs - outputs) ** 2, dim=tuple(range(1, inputs.dim())))
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
            loss = recon_loss + kl_loss
            threshold = adaptive_thresholding(loss)

            for idx in range(inputs.size(0)):
                if loss[idx] >= threshold:
                    decoded_row = dataset.inverse_transform(inputs[idx].cpu().numpy())  # Adjust as per your dataset

                    aircraft_registration_list = [key.split('_')[-1] for key, value in decoded_row.items() if
                                                  'AIRCRAFT_REGISTRATION' in key and value == 1]
                    arr_airport_list = [key.split('_')[-1] for key, value in decoded_row.items() if
                                        'ARR_AIRPORT' in key and value == 1]
                    dep_airport_list = [key.split('_')[-1] for key, value in decoded_row.items() if
                                        'DEP_AIRPORT' in key and value == 1]

                    aircraft_registration = aircraft_registration_list[0] if aircraft_registration_list else None
                    arr_airport = arr_airport_list[0] if arr_airport_list else None
                    dep_airport = dep_airport_list[0] if dep_airport_list else None

                    anomalies_df.loc[len(anomalies_df)] = [aircraft_registration, arr_airport, dep_airport]

                    # Extract relevant features for LimeTabularExplainer
                    data_row = np.array(list(decoded_row.values())).reshape(1, -1)  # Reshape for LimeTabularExplainer

                    explained_instance = explainer.explain_instance(data_row[0], predict_fn,
                                                                    num_features=len(feature_names))

                    # Get the list of feature names and weights as tuples
                    explanation_features = explained_instance.as_list()

                    # Initialize variables to store extracted values
                    aircraft_registration = None
                    arr_airport = None
                    dep_airport = None

                    # Extract values based on feature names
                    for feature, weight in explanation_features:
                        if 'AIRCRAFT_REGISTRATION' in feature:
                            aircraft_registration = feature.split('_')[-1], round(weight, 2)
                        elif 'ARR_AIRPORT' in feature:
                            arr_airport = feature.split('_')[-1], round(weight, 2)
                        elif 'DEP_AIRPORT' in feature:
                            dep_airport = feature.split('_')[-1], round(weight, 2)

                    # Store individual components for explanation DataFrame
                    explanations.append({
                        'A/C Reg': aircraft_registration[0].split()[0] if aircraft_registration else None,
                        'A/C value': aircraft_registration[1] if aircraft_registration else None,
                        'Arr Airport': arr_airport[0].split()[0] if arr_airport else None,
                        'Arr value': arr_airport[1] if arr_airport else None,
                        'Dep Airport': dep_airport[0].split()[0] if dep_airport else None,
                        'Dep value': dep_airport[1] if dep_airport else None
                    })

    # Create DataFrame for explanations
    explanations_df = pd.DataFrame(explanations, columns=['A/C Reg', 'A/C value',
                                                          'Arr Airport', 'Arr value',
                                                          'Dep Airport', 'Dep value'])

    # Print anomalies and matched explanations for debugging
    print(explanations_df)
    print(anomalies_df)


def predict_fn(x):
    return autoencoder(torch.tensor(x).float())[0].detach().numpy()


def adaptive_thresholding(loss, k=multiplier):
    mean = torch.mean(loss)
    std = torch.std(loss)
    threshold = mean + k * std
    return threshold
