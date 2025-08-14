import os
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def load_data(base_dir="namadataset_preprocessing", csv_name="Iris_preprocessed.csv"):
    csv_path = os.path.join(base_dir, csv_name)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset tidak ditemukan: {csv_path}")
    df = pd.read_csv(csv_path)
    X = df.drop(columns=["Species"])
    y = df["Species"]
    return X, y

def main():
    mlflow.set_experiment("Iris_Classification_eka-fanya")
    mlflow.sklearn.autolog()

    # Load dataset siap latih
    X, y = load_data()

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    with mlflow.start_run(run_name="RandomForest_Basic"):
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        acc = accuracy_score(y_test, y_pred)
        print(f"Akurasi Model: {acc:.4f}")
        print("\nLaporan Klasifikasi:")
        print(classification_report(y_test, y_pred))

        # Simpan scaler sebagai artefak tambahan
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            scaler_path = os.path.join(d, "scaler.joblib")
            joblib.dump(scaler, scaler_path)
            mlflow.log_artifact(scaler_path, artifact_path="preprocess")

if __name__ == "__main__":
    main()
