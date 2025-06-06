# model_fertilizer.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
data = pd.read_csv('data/Fertilizer.csv')

# Features and target
X = data[['Nitrogen', 'Potassium', 'Phosphorous']]
y = data['Fertilizer Name']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save the trained model
joblib.dump(model, 'data/trained_fertilizer_model.pkl')

print("Fertilizer recommendation model trained and saved successfully!")