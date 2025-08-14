import argparse, os, joblib, mlflow, mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--test_size", type=float, default=0.2)
    p.add_argument("--random_state", type=int, default=42)
    return p.parse_args()

def load_data(base="namadataset_preprocessing", csv="Iris_preprocessed.csv"):
    path = os.path.join(base, csv)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset tidak ditemukan: {path}")
    df = pd.read_csv(path)
    return df.drop(columns=["Species"]), df["Species"]

if __name__ == "__main__":
    args = parse()
    mlflow.set_experiment("Iris_Classification_eka-fanya_CI")
    mlflow.sklearn.autolog()

    X, y = load_data()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=args.test_size,
                                          random_state=args.random_state, stratify=y)
    sc = StandardScaler()
    Xtr_s = sc.fit_transform(Xtr); Xte_s = sc.transform(Xte)

    with mlflow.start_run(run_name="CI_RandomForest"):
        model = RandomForestClassifier(n_estimators=120, max_depth=5, random_state=args.random_state)
        model.fit(Xtr_s, ytr)
        pred = model.predict(Xte_s)
        acc = accuracy_score(yte, pred)
        mlflow.log_metric("accuracy", acc)
        print("Akurasi:", acc)
