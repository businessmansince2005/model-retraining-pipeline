import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import joblib
import mlflow

os.makedirs("models/latest", exist_ok=True)

def load_data():
    path = os.path.join("data", "new_data.csv")
    if not os.path.exists(path):
        rng = np.random.RandomState(42)
        X = pd.DataFrame({
            "feature1": rng.normal(size=500),
            "feature2": rng.normal(size=500),
        })
        y = (X["feature1"] * 0.8 + X["feature2"] * -0.6 + rng.normal(scale=0.5, size=500) > 0).astype(int)
        df = X.copy()
        df["target"] = y
        return df
    return pd.read_csv(path)

def train():
    df = load_data()
    X = df.drop(columns=["target"])
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with mlflow.start_run():
        model = LogisticRegression(max_iter=500)
        model.fit(X_train, y_train)
        y_pred = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred)

        mlflow.log_metric("roc_auc", float(auc))

        out_path = os.path.join("models", "latest", "model.pkl")
        joblib.dump(model, out_path)
        mlflow.log_artifact(out_path, artifact_path="model")

        with open(os.path.join("models", "latest", "VERSION.txt"), "w") as f:
            f.write(f"ROC_AUC={auc:.4f}\n")

        print(f"Saved latest model to {out_path} with ROC_AUC={auc:.4f}")

if __name__ == "__main__":
    train()