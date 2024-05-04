import joblib
import numpy as np
import pandas as pd
import sys

# Load the model from the .joblib file
loaded_model = joblib.load('crop_recommendation_model.joblib')


def predict_top_three_crops(new_data):
    # Make predictions
    probabilities = loaded_model.predict_proba(new_data.values)

    # Get the class labels
    classes = loaded_model.classes_

    # Get the top three crop classes with highest probabilities
    top_three_indices = probabilities.argsort()[:, -3:][:, ::-1]
    top_three_probs = probabilities[np.arange(len(probabilities)), top_three_indices]
    top_three_crops = [classes[i] for i in top_three_indices]

    return top_three_crops, top_three_probs

if len(sys.argv) > 1:
    # Example of making predictions on new data from command line arguments
    new_data = pd.DataFrame({
        'N': [float(sys.argv[1])],  # Example value for N
        'P': [float(sys.argv[2])],  # Example value for P
        'K': [float(sys.argv[3])],  # Example value for K
        'temperature': [float(sys.argv[4])],  # Example value for temperature
        'humidity': [float(sys.argv[5])],  # Example value for humidity
        'ph': [float(sys.argv[6])],  # Example value for ph
        'rainfall': [float(sys.argv[7])],  # Example value for rainfall
        'Total_Nutrients': [float(sys.argv[8])],  # Example value for Total_Nutrients
        'Temperature_Humidity': [float(sys.argv[9])],  # Example value for Temperature_Humidity
        'Log_Rainfall': [float(sys.argv[10])]  # Example value for Log_Rainfall
    })
else:
    new_data = pd.DataFrame({
        'N': [40],  # Example value for N
        'P': [60],  # Example value for P
        'K': [40],  # Example value for K
        'temperature': [25],  # Example value for temperature
        'humidity': [80],  # Example value for humidity
        'ph': [6.5],  # Example value for ph
        'rainfall': [200],  # Example value for rainfall
        'Total_Nutrients': [150],  # Example value for Total_Nutrients
        'Temperature_Humidity': [1700],  # Example value for Temperature_Humidity
        'Log_Rainfall': [5]  # Example value for Log_Rainfall
    })


# Make predictions for the top three crops
top_three_crops, top_three_probs = predict_top_three_crops(new_data)
print("Top three predicted crops:", top_three_crops[0])
print("Corresponding probabilities:", top_three_probs[0])
