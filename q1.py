import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the dataset
data = pd.read_csv("../../../../../Downloads/Crop_Dataset.csv")

# Split the dataset into features (X) and target variable (y)
X = data.drop(['Label', 'Label_Encoded'], axis=1)
y = data['Label']

# Scale numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train the model
model = RandomForestClassifier()
model.fit(X_scaled, y)

# Save the trained model to a .joblib file
joblib.dump(model, 'crop_recommendation_model.joblib')
