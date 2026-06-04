import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import joblib
import os

print("Generating synthetic Customer Churn dataset...")
# Synthetic data generation for demonstration
np.random.seed(42)
n_samples = 2000

# Features: MonthlyCharges, Tenure (months), TotalCharges, ContractType (0: Month-to-month, 1: One year, 2: Two year)
tenure = np.random.randint(1, 73, size=n_samples)
monthly_charges = np.random.uniform(20.0, 120.0, size=n_samples)
total_charges = tenure * monthly_charges * np.random.uniform(0.9, 1.1, size=n_samples)
contract_type = np.random.choice([0, 1, 2], size=n_samples, p=[0.5, 0.3, 0.2])

# Churn logic: High monthly charges, low tenure, and month-to-month contracts increase churn probability
churn_prob = (
    (monthly_charges > 80).astype(int) * 0.3 + 
    (tenure < 12).astype(int) * 0.4 + 
    (contract_type == 0).astype(int) * 0.3
)
churn = (np.random.rand(n_samples) < churn_prob).astype(int)

df = pd.DataFrame({
    'Tenure': tenure,
    'MonthlyCharges': monthly_charges,
    'TotalCharges': total_charges,
    'ContractType': contract_type,
    'Churn': churn
})

X = df.drop('Churn', axis=1)
y = df['Churn']

# Preprocessing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model Training
print("Training XGBoost Classifier...")
model = xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
model.fit(X_train_scaled, y_train)

# Evaluation
y_pred = model.predict(X_test_scaled)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Save the model and scaler
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/xgboost_churn_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
print("Model and scaler saved to 'models/' directory.")
