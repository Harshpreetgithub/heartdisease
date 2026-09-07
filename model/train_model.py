"""
train_model.py
----------------
Reproduces the preprocessing + model-selection pipeline from
`Heart_Disease_Prediction_-_Classification.ipynb` and serializes the
winning model (Random Forest) plus the scaler and the exact feature
column order, so the Vercel API function can make predictions that
are 100% consistent with how the model was trained.

Run locally with:
    python model/train_model.py

Outputs (written to model/):
    model.pkl           -> trained RandomForestClassifier
    scaler.pkl           -> fitted StandardScaler
    feature_columns.json -> exact column order expected by the model
    metrics.json          -> accuracy / F1 / CV score for the record
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "..", "data", "heart.csv")


def load_and_preprocess(path=DATA_PATH):
    dataset = pd.read_csv(path)

    # Same label encoding as the notebook: Sex (F=0, M=1), ExerciseAngina (N=0, Y=1)
    label_enc = LabelEncoder()
    dataset["Sex"] = label_enc.fit_transform(dataset["Sex"])
    dataset["ExerciseAngina"] = LabelEncoder().fit_transform(dataset["ExerciseAngina"])

    # Same one-hot encoding as the notebook
    dataset = pd.get_dummies(
        dataset, columns=["ChestPainType", "RestingECG", "ST_Slope"], drop_first=True
    )

    return dataset


def main():
    dataset = load_and_preprocess()

    X = dataset.drop("HeartDisease", axis="columns")
    y = dataset["HeartDisease"].values
    feature_columns = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y, test_size=0.2, random_state=0, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Random Forest was the best-performing model in the notebook's comparison
    model = RandomForestClassifier(n_estimators=100, random_state=0)
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    train_acc = accuracy_score(y_train, y_pred_train) * 100
    test_acc = accuracy_score(y_test, y_pred_test) * 100
    test_f1 = f1_score(y_test, y_pred_test)
    cv_scores = cross_val_score(model, X_train, y_train, cv=10)

    print(f"Training accuracy: {train_acc:.2f}%")
    print(f"Testing accuracy:  {test_acc:.2f}%")
    print(f"Test F1-score:     {test_f1:.4f}")
    print(f"CV accuracy:       {cv_scores.mean() * 100:.2f}%")
    print()
    print(classification_report(y_test, y_pred_test, target_names=["No Disease", "Disease"]))

    joblib.dump(model, os.path.join(HERE, "model.pkl"))
    joblib.dump(scaler, os.path.join(HERE, "scaler.pkl"))

    with open(os.path.join(HERE, "feature_columns.json"), "w") as f:
        json.dump(feature_columns, f, indent=2)

    with open(os.path.join(HERE, "metrics.json"), "w") as f:
        json.dump(
            {
                "train_accuracy": round(train_acc, 2),
                "test_accuracy": round(test_acc, 2),
                "test_f1": round(test_f1, 4),
                "cv_accuracy": round(cv_scores.mean() * 100, 2),
            },
            f,
            indent=2,
        )

    print("\nSaved model.pkl, scaler.pkl, feature_columns.json, metrics.json to model/")


if __name__ == "__main__":
    main()
