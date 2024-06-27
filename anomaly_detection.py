import pandas as pd
import torch
import numpy as np
from lime.lime_tabular import LimeTabularExplainer
from hyperparameter import multiplier

def detect_anomalies(test_dataloader, autoencoder, dataset):
    autoencoder.eval()
    anomalies_df = pd.DataFrame(columns=['A/C Registration', 'Arr Airport',
                                         'Dep Airport', 'A/C weight',
                                         'Arr weight', 'Dep weight'])
    explanations = []

    feature_names = dataset.columns
    data_array = dataset.data.values
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
                    decoded_row = dataset.inverse_transform(inputs[idx].cpu().numpy())

                    aircraft_registration_list = [key.split('_')[-1] for key, value in decoded_row.items() if
                                                  'AIRCRAFT_REGISTRATION' in key and value == 1]
                    arr_airport_list = [key.split('_')[-1] for key, value in decoded_row.items() if
                                        'ARR_AIRPORT' in key and value == 1]
                    dep_airport_list = [key.split('_')[-1] for key, value in decoded_row.items() if
                                        'DEP_AIRPORT' in key and value == 1]

                    aircraft_registration = aircraft_registration_list[0] if aircraft_registration_list else None
                    arr_airport = arr_airport_list[0] if arr_airport_list else None
                    dep_airport = dep_airport_list[0] if dep_airport_list else None
                    anomalies_df.loc[len(anomalies_df)] = [aircraft_registration, arr_airport,
                                                           dep_airport, None, None, None]

                    # Extract relevant features for LimeTabularExplainer
                    data_row = np.array(list(decoded_row.values())).reshape(1, -1)  # Reshape for LimeTabularExplainer
                    explained_instance = explainer.explain_instance(data_row[0], predict_fn,
                                                                    num_features=len(feature_names))

                    # Get the list of feature names and weights
                    explanation_features = explained_instance.as_list()

                    for feature, weight in explanation_features:
                        explanations.append({
                            'Features': feature,
                            'Weight': weight
                        })

    explanations_df = pd.DataFrame(explanations, columns=['Features', 'Weight'])

    # Check if anomalies match features and add corresponding weight
    for index, row in anomalies_df.iterrows():
        for feature, weight in zip(explanations_df['Features'], explanations_df['Weight']):
            if row['A/C Registration'] is not None and 'AIRCRAFT_REGISTRATION' in feature \
                    and row['A/C Registration'] in feature:
                anomalies_df.loc[index, 'A/C weight'] = weight
            elif row['Arr Airport'] is not None and 'ARR_AIRPORT' in feature and row['Arr Airport'] in feature:
                anomalies_df.loc[index, 'Arr weight'] = weight
            elif row['Dep Airport'] is not None and 'DEP_AIRPORT' in feature and row['Dep Airport'] in feature:
                anomalies_df.loc[index, 'Dep weight'] = weight

    return(anomalies_df)


def predict_fn(x):
    return autoencoder(torch.tensor(x).float())[0].detach().numpy()


def adaptive_thresholding(loss, k=multiplier):
    mean = torch.mean(loss)
    std = torch.std(loss)
    threshold = mean + k * std
    return threshold
