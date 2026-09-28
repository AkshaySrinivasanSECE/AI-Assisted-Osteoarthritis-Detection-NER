from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "full_indicators_summary.csv"
MODELS_DIR = ROOT_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)


# --------------------------------------------------
# 1. LOAD DATASET
# --------------------------------------------------

df = pd.read_csv(
    DATA_PATH,
    encoding="gb18030"
)

# Rename difficult column names
df = df.rename(columns={
    "30～ JPR": "JPR_30",
    "45～JPR": "JPR_45",
    "60～JPR": "JPR_60"
})


# --------------------------------------------------
# 2. SELECT FEATURES
# --------------------------------------------------

features = [
    "age",
    "gender",
    "BMI",
    "VAS score",
    "JPR_30",
    "JPR_45",
    "JPR_60"
]

X = df[features]

# 0 = Healthy
# 1 = Knee Osteoarthritis
y = df["group"]


print("Dataset shape:", df.shape)

print("\nTarget distribution:")
print(y.value_counts())

print("\nFeatures used:")
print(features)


# --------------------------------------------------
# 3. DEFINE MODELS FROM YOUR SYLLABUS
# --------------------------------------------------

models = {

    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=1000,
            random_state=42
        ))
    ]),

    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(
            n_neighbors=5
        ))
    ]),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=4,
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=5,
        random_state=42
    ),

    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(
            kernel="rbf",
            probability=True,
            random_state=42
        ))
    ])
}


# --------------------------------------------------
# 4. CROSS VALIDATION
# --------------------------------------------------

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scoring = [
    "accuracy",
    "precision",
    "recall",
    "f1"
]

results = []


for name, model in models.items():

    scores = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=scoring
    )

    accuracy = scores["test_accuracy"].mean()
    precision = scores["test_precision"].mean()
    recall = scores["test_recall"].mean()
    f1 = scores["test_f1"].mean()

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    })


# --------------------------------------------------
# 5. SHOW RESULTS
# --------------------------------------------------

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="F1",
    ascending=False
)

print("\nMODEL COMPARISON")
print("=" * 70)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.3f}"
    )
)


# --------------------------------------------------
# 6. SELECT BEST MODEL
# --------------------------------------------------

best_model_name = results_df.iloc[0]["Model"]

best_model = models[best_model_name]

print("\nBest model:", best_model_name)


# --------------------------------------------------
# 7. TRAIN BEST MODEL ON FULL DATASET
# --------------------------------------------------

best_model.fit(X, y)


# --------------------------------------------------
# 8. SAVE MODEL
# --------------------------------------------------

joblib.dump(
    best_model,
    MODELS_DIR / "oa_model.pkl"
)

joblib.dump(
    features,
    MODELS_DIR / "features.pkl"
)


print("\nModel saved successfully:")
print("models/oa_model.pkl")
print("models/features.pkl")