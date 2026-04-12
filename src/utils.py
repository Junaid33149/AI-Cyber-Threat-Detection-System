import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

def preprocess_data(data):

    # ==============================
    # CLEAN COLUMN NAMES
    # ==============================
    data = data.copy()
    data.columns = data.columns.str.strip()

    # ==============================
    # DROP TIMESTAMP IF EXISTS
    # ==============================
    if "Timestamp" in data.columns:
        data = data.drop("Timestamp", axis=1)

    # ==============================
    # HANDLE INF VALUES (IMPORTANT FIX)
    # ==============================
    data.replace([np.inf, -np.inf], np.nan, inplace=True)

    # ==============================
    # DROP MISSING VALUES
    # ==============================
    data = data.dropna()

    # ==============================
    # LABEL COLUMN FIX
    # ==============================
    if "Attack_Label" in data.columns:
        label_col = "Attack_Label"
    elif "Label" in data.columns:
        label_col = "Label"
    else:
        print("❌ ERROR: No label column found")
        print("Columns:", data.columns)
        exit()

    # ==============================
    # CONVERT LABEL → 0/1
    # ==============================
    data[label_col] = data[label_col].apply(
        lambda x: 0 if str(x).lower() == "benign" else 1
    )

    # ==============================
    # ENCODE CATEGORICAL FEATURES
    # ==============================
    le = LabelEncoder()
    for col in data.select_dtypes(include=['object']).columns:
        data.loc[:, col] = le.fit_transform(data[col])

    # ==============================
    # SPLIT FEATURES & LABEL
    # ==============================
    X = data.drop(label_col, axis=1)
    y = data[label_col]

    # ==============================
    # FINAL SAFETY CHECK (IMPORTANT)
    # ==============================
    X = X.astype(float)

    # ==============================
    # SCALE FEATURES
    # ==============================
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler, X.columns