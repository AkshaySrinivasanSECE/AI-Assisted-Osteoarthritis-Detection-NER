from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "full_indicators_summary.csv"
RESULTS_DIR = ROOT / "docs" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = [
    "age",
    "gender",
    "BMI",
    "VAS score",
    "JPR_30",
    "JPR_45",
    "JPR_60",
]


def load_dataset():
    df = pd.read_csv(DATA_PATH, encoding="gb18030")
    df = df.rename(columns={
        "30～ JPR": "JPR_30",
        "45～JPR": "JPR_45",
        "60～JPR": "JPR_60",
    })
    return df


def build_models():
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=1000, random_state=42)),
        ]),
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("model", KNeighborsClassifier(n_neighbors=5)),
        ]),
        "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42),
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(kernel="rbf", probability=True, random_state=42)),
        ]),
    }


def save_model_comparison(results_df):
    fig, ax = plt.subplots(figsize=(10, 6))
    metrics = ["Accuracy", "Precision", "Recall", "F1"]
    x = range(len(results_df))
    width = 0.18
    colors = ["#3b82f6", "#10b981", "#f59e0b", "#ef4444"]

    for idx, metric in enumerate(metrics):
        ax.bar(
            [i + idx * width for i in x],
            results_df[metric].tolist(),
            width=width,
            label=metric,
            color=colors[idx],
            alpha=0.9,
        )

    ax.set_title("Model Comparison Across Stratified 5-Fold CV")
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_xticks([i + 1.5 * width for i in x])
    ax.set_xticklabels(results_df["Model"], rotation=20, ha="right")
    ax.legend(loc="best")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "model_comparison.png", dpi=200)
    plt.close(fig)


def save_class_distribution(y):
    fig, ax = plt.subplots(figsize=(7, 5))
    counts = y.value_counts().sort_index()
    labels = ["Healthy", "KOA"]
    ax.bar(labels, counts.values, color=["#2a9d8f", "#e76f51"])
    ax.set_title("Dataset Class Distribution")
    ax.set_ylabel("Subjects")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "class_distribution.png", dpi=200)
    plt.close(fig)


def save_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=["Healthy", "Knee OA"]).plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title("Confusion Matrix (Stratified 5-Fold CV)")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=200)
    plt.close(fig)


def save_feature_importance(model, feature_names):
    if not hasattr(model, "feature_importances_"):
        return

    importances = model.feature_importances_
    sorted_idx = importances.argsort()[::-1]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh([feature_names[i] for i in sorted_idx], importances[sorted_idx], color="#1d4ed8")
    ax.invert_yaxis()
    ax.set_title("Random Forest Feature Importance")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "feature_importance.png", dpi=200)
    plt.close(fig)


def main():
    df = load_dataset()
    X = df[FEATURES]
    y = df["group"]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = ["accuracy", "precision", "recall", "f1"]
    results = []

    for name, model in build_models().items():
        scores = cross_validate(model, X, y, cv=cv, scoring=scoring)
        results.append({
            "Model": name,
            "Accuracy": scores["test_accuracy"].mean(),
            "Precision": scores["test_precision"].mean(),
            "Recall": scores["test_recall"].mean(),
            "F1": scores["test_f1"].mean(),
        })

    results_df = pd.DataFrame(results).sort_values(by="F1", ascending=False).reset_index(drop=True)
    print("\nModel comparison summary")
    print(results_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    save_model_comparison(results_df)

    best_model_name = results_df.iloc[0]["Model"]
    best_model = build_models()[best_model_name]
    best_model.fit(X, y)

    y_pred = cross_val_predict(best_model, X, y, cv=cv, method="predict")
    print("\nClassification report")
    print(classification_report(y, y_pred, target_names=["Healthy", "Knee OA"]))

    save_confusion_matrix(y, y_pred)
    save_class_distribution(y)
    save_feature_importance(best_model, FEATURES)

    print("\nSaved evaluation files:")
    for file_name in [
        "confusion_matrix.png",
        "class_distribution.png",
        "feature_importance.png",
        "model_comparison.png",
    ]:
        print(f"- {RESULTS_DIR / file_name}")


if __name__ == "__main__":
    main()
