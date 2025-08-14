import os, tempfile, joblib, mlflow, mlflow.sklearn
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
from mlflow.models import infer_signature

DATA_DIR = "namadataset_preprocessing"
CSV_NAME = "Iris_preprocessed.csv"
TARGET   = "Species"

def load_data():
    path = os.path.join(DATA_DIR, CSV_NAME)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset tidak ditemukan: {path}")
    df = pd.read_csv(path)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return X, y

def plot_confusion(cm, classes, out_png):
    fig = plt.figure(figsize=(4,3))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix"); plt.colorbar()
    ticks = np.arange(len(classes))
    plt.xticks(ticks, classes, rotation=45); plt.yticks(ticks, classes)
    thr = cm.max()/2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > thr else "black")
    plt.tight_layout(); plt.ylabel("True"); plt.xlabel("Pred")
    fig.savefig(out_png, bbox_inches="tight"); plt.close(fig)

if __name__ == "__main__":
    # === TIDAK pakai autolog (wajib Skilled) ===
    mlflow.sklearn.autolog(disable=True)
    mlflow.set_experiment("Iris_Classification_eka-fanya_TUNING")

    # 1) Load data
    X, y = load_data()

    # 2) Split & (re)scale ringan (aman meski file sudah scaled)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)

    # 3) GridSearchCV
    grid = {
        "n_estimators": [80, 120, 160],
        "max_depth": [3, 5, None],
        "min_samples_split": [2, 4]
    }
    base = RandomForestClassifier(random_state=42)
    gs = GridSearchCV(base, grid, cv=3, n_jobs=-1)
    gs.fit(Xtr_s, ytr)
    best = gs.best_estimator_

    # 4) Eval
    ypred = best.predict(Xte_s)
    acc  = accuracy_score(yte, ypred)
    prec = precision_score(yte, ypred, average="macro")
    rec  = recall_score(yte, ypred, average="macro")
    f1   = f1_score(yte, ypred, average="macro")

    # 5) Manual logging ke MLflow
    with mlflow.start_run(run_name="RF_GridSearch_manual"):
        # params
        for k, v in gs.best_params_.items():
            mlflow.log_param(k, v)
        mlflow.log_param("cv", 3)
        mlflow.log_param("scaler", "StandardScaler")

        # metrics (lebih dari autolog: precision/recall macro)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision_macro", prec)
        mlflow.log_metric("recall_macro", rec)
        mlflow.log_metric("f1_macro", f1)

        # artifacts: confusion matrix + scaler
        cm = confusion_matrix(yte, ypred)
        with tempfile.TemporaryDirectory() as d:
            cm_png = os.path.join(d, "confusion_matrix.png")
            plot_confusion(cm, classes=sorted(np.unique(y)), out_png=cm_png)
            mlflow.log_artifact(cm_png, artifact_path="figures")

            sc_path = os.path.join(d, "scaler.joblib")
            joblib.dump(scaler, sc_path)
            mlflow.log_artifact(sc_path, artifact_path="preprocess")

        # model with signature & input_example (best practice)
        input_example = Xte_s[:1]
        signature = infer_signature(Xtr_s, best.predict(Xtr_s))
        mlflow.sklearn.log_model(best, artifact_path="model",
                                 input_example=input_example, signature=signature)

    print(f"✅ Tuning selesai. Acc={acc:.4f}  Prec={prec:.4f}  Rec={rec:.4f}  F1={f1:.4f}")
