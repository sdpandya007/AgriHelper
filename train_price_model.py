import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# Debug: Print current working directory
print("Current working directory:", os.getcwd())

# Load Dataset
try:
    # Load dataset using read_csv if it's a CSV file
    data = pd.read_csv("data/Agmarknet_Price_Report.csv")  # Replace with your file path
    print("Columns in the dataset:", data.columns.tolist())  # Debug: Print column names
except FileNotFoundError:
    print("Error: The file 'Agmarknet_Price_Report.csv' was not found. Please check the file path.")
    exit()

# Clean column names (remove leading/trailing spaces)
data.columns = data.columns.str.strip()

# Preprocess Data
label_encoders = {}
required_columns = ["District name", "Market name", "Commodity", "Variety", "Grade", "Price date", "Modal price (Rs./Quintal)"]

# Validate required columns
missing_columns = [col for col in required_columns if col not in data.columns]
if missing_columns:
    print(f"Error: The following required columns are missing in the dataset: {missing_columns}")
    exit()

# Encode categorical variables
for column in ["District name", "Market name", "Commodity", "Variety", "Grade"]:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Convert 'Price date' to ordinal (days since a reference date)
data['Price date'] = pd.to_datetime(data['Price date'], errors='coerce')  # Handle invalid dates
data['Price date'] = data['Price date'].map(lambda x: x.toordinal() if pd.notnull(x) else None)

# Drop rows with missing values
data = data.dropna(subset=["Price date", "Modal price (Rs./Quintal)"])

# Define Features and Target
X = data[["District name", "Market name", "Commodity", "Variety", "Grade", "Price date"]]
y = data["Modal price (Rs./Quintal)"]

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save Model and Encoders
joblib.dump(model, "data/trained_price_model.pkl")
joblib.dump(label_encoders, "data/label_encoders.pkl")

print("Model and encoders saved successfully!")