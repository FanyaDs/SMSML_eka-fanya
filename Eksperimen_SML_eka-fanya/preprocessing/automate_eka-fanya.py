import os
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess(
    input_csv="../namadataset_raw/Iris.csv",
    out_dir="/content/drive/MyDrive/Colab Notebooks/SMSML_eka-fanya/Membangun_model/namadataset_preprocessing",
    drop_cols=("Id",),
    label_col="Species"
):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(input_csv)

    for c in drop_cols:
        if c in df.columns:
            df = df.drop(columns=[c])

    df = df.drop_duplicates().reset_index(drop=True)

    if df[label_col].dtype == "O":
        le = LabelEncoder()
        df[label_col] = le.fit_transform(df[label_col])
    else:
        le = LabelEncoder()
        le.classes_ = pd.Index(sorted(df[label_col].unique()))

    X = df.drop(columns=[label_col])
    y = df[label_col]

    scaler = StandardScaler()
    scaler.fit(X)

    out_csv = os.path.join(out_dir, "Iris_preprocessed.csv")
    df_ready = pd.concat([X.reset_index(drop=True), y.reset_index(drop=True).rename(label_col)], axis=1)
    df_ready.to_csv(out_csv, index=False)

    joblib.dump(scaler, os.path.join(out_dir, "scaler.joblib"))
    joblib.dump(le,     os.path.join(out_dir, "label_encoder.joblib"))

    print("✅ Saved:", out_csv)
    print("✅ Saved:", os.path.join(out_dir, "scaler.joblib"))
    print("✅ Saved:", os.path.join(out_dir, "label_encoder.joblib"))

if __name__ == "__main__":
    preprocess()
