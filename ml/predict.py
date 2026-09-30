import pandas as pd
import joblib

df = pd.read_csv(
    "data/full_indicators_summary.csv",
    encoding="utf-8-sig"
)

model = joblib.load("models/oa_model.pkl")
features = joblib.load("models/features.pkl")

patient = df.iloc[[0]]

X_patient = patient[features]

prediction = model.predict(X_patient)[0]
probabilities = model.predict_proba(X_patient)[0]

print("\nPatient data:")
print(X_patient)

print("\nActual group:", patient["group"].iloc[0])
print("Predicted group:", prediction)

print("\nHealthy probability:", round(probabilities[0] * 100, 2), "%")
print("OA probability:", round(probabilities[1] * 100, 2), "%")

if prediction == 1:
    print("\nResult: OA RISK DETECTED")
else:
    print("\nResult: LOW OA RISK")