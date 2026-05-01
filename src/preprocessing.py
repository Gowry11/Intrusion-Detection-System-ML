"""
Preprocessing pipeline for NSL-KDD dataset.
Handles loading, encoding, scaling, and train-test split.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

from .utils import DATA_DIR, RANDOM_STATE, find_dataset_file, ensure_dirs


# NSL-KDD column names (41 features + label; last column in test = difficulty)
NSL_KDD_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "label"
]

# Map attack types to categories for multi-class
ATTACK_MAPPING = {
    "normal": "Normal",
    "back": "DoS", "land": "DoS", "neptune": "DoS", "pod": "DoS", "smurf": "DoS", "teardrop": "DoS",
    "ipsweep": "Probe", "nmap": "Probe", "portsweep": "Probe", "satan": "Probe", "saint": "Probe", "mscan": "Probe",
    "buffer_overflow": "U2R", "loadmodule": "U2R", "perl": "U2R", "rootkit": "U2R",
    "ftp_write": "R2L", "guess_passwd": "R2L", "imap": "R2L", "multihop": "R2L",
    "phf": "R2L", "spy": "R2L", "warezclient": "R2L", "warezmaster": "R2L",
}


def _load_csv(path, sep=None):
    """Load CSV/TXT with flexible separator (no header)."""
    for s in (sep or [",", ";", "\t"]):
        try:
            df = pd.read_csv(path, sep=s, header=None, low_memory=False)
            if df.shape[1] >= 42:
                return df
        except Exception:
            continue
    raise ValueError(f"Could not parse {path}")


def _load_single_csv(path):
    """Load combined NSL-KDD CSV with header (e.g. nsl-kdd.xlsx.csv)."""
    df = pd.read_csv(path, low_memory=False)
    # Map attack_class -> label
    if "attack_class" in df.columns and "label" not in df.columns:
        df = df.rename(columns={"attack_class": "label"})
    # Map is_hot_login -> is_host_login for consistency
    if "is_hot_login" in df.columns and "is_host_login" not in df.columns:
        df = df.rename(columns={"is_hot_login": "is_host_login"})
    # Drop extra columns (num_learners, difficulty, etc.)
    for col in ["num_learners", "difficulty"]:
        if col in df.columns:
            df = df.drop(columns=[col])
    if "label" not in df.columns:
        raise ValueError("CSV must have 'label' or 'attack_class' column")
    return df


def load_dataset(data_dir=None):
    """
    Load NSL-KDD dataset.
    Supports: (1) Single combined CSV (nsl-kdd.xlsx.csv) in project root or data/
              (2) Separate KDDTrain+ and KDDTest+ files in data/
    """
    data_dir = Path(data_dir) if data_dir else DATA_DIR
    ensure_dirs()

    # Prefer combined file (e.g. nsl-kdd.xlsx.csv in project root)
    combined_path = find_dataset_file(
        ["nsl-kdd.xlsx.csv", "nsl-kdd.csv", "NSL-KDD.csv"],
        data_dir,
        also_project_root=True,
    )
    if combined_path is not None:
        df = _load_single_csv(combined_path)
        from sklearn.model_selection import train_test_split
        try:
            df_train, df_test = train_test_split(
                df, test_size=0.2, stratify=df["label"], random_state=RANDOM_STATE
            )
        except ValueError:
            # Rare classes with <2 samples; use random split
            df_train, df_test = train_test_split(
                df, test_size=0.2, random_state=RANDOM_STATE
            )
        df_train = df_train.reset_index(drop=True)
        df_test = df_test.reset_index(drop=True)
        return df_train, df_test

    # Fall back to separate train/test files
    train_path = find_dataset_file(
        ["KDDTrain+.csv", "KDDTrain+.txt", "Train.csv", "train.csv"], data_dir
    )
    test_path = find_dataset_file(
        ["KDDTest+.csv", "KDDTest+.txt", "Test.csv", "test.csv"], data_dir
    )

    if not train_path:
        raise FileNotFoundError(
            f"No dataset found. Place nsl-kdd.xlsx.csv in project root, or "
            f"KDDTrain+.csv / KDDTest+.csv in {data_dir}."
        )
    if not test_path:
        raise FileNotFoundError(
            f"No test file found. Place KDDTest+.csv in {data_dir}."
        )

    df_train = _load_csv(train_path)
    df_test = _load_csv(test_path)

    n_cols_train = df_train.shape[1]
    n_cols_test = df_test.shape[1]
    if n_cols_train == 43:
        df_train.columns = NSL_KDD_COLUMNS + ["difficulty"]
    else:
        df_train.columns = NSL_KDD_COLUMNS[:n_cols_train]
    if n_cols_test == 43:
        df_test.columns = NSL_KDD_COLUMNS + ["difficulty"]
    else:
        df_test.columns = NSL_KDD_COLUMNS[:n_cols_test]

    for df in [df_train, df_test]:
        if "difficulty" in df.columns:
            df.drop(columns=["difficulty"], inplace=True, errors="ignore")

    return df_train, df_test


def _map_labels(df):
    """Map raw labels to binary and multi-class."""
    labels = df["label"].astype(str).str.strip().str.lower().str.rstrip(".")
    binary = labels.apply(lambda x: "attack" if x != "normal" else "normal")
    multiclass = labels.apply(
        lambda x: ATTACK_MAPPING.get(x, "Normal" if x == "normal" else "Other")
    )
    return binary, multiclass


def remove_redundant_columns(df):
    """Remove redundant columns (e.g., constant or near-constant)."""
    # num_outbound_cmds is always 0 in NSL-KDD
    redundant = ["num_outbound_cmds"]
    return df.drop(columns=[c for c in redundant if c in df.columns], errors="ignore")


def preprocess(
    data_dir=None,
    test_size=0.2,
    task="multiclass",
    save_artifacts=True,
    output_dir=None,
):
    """
    Full preprocessing pipeline.

    Parameters
    ----------
    data_dir : str or Path, optional
        Path to data folder.
    test_size : float
        Fraction for test set (when combining train+test for stratified split).
    task : str
        'binary' or 'multiclass'
    save_artifacts : bool
        Whether to save scaler, encoders, and processed data.
    output_dir : Path, optional
        Where to save artifacts.

    Returns
    -------
    dict with X_train, X_test, y_train, y_test, feature_names, preprocessors, etc.
    """
    data_dir = Path(data_dir) if data_dir else DATA_DIR
    output_dir = Path(output_dir) if output_dir else DATA_DIR
    ensure_dirs()

    df_train, df_test = load_dataset(data_dir)

    # Combine for stratified split (ensure both train/test in split)
    df = pd.concat([df_train, df_test], ignore_index=True)

    df = remove_redundant_columns(df)

    # Create binary and multi-class labels
    binary_labels, multiclass_labels = _map_labels(df)
    df["binary_label"] = binary_labels
    df["multiclass_label"] = multiclass_labels

    y = df["binary_label"] if task == "binary" else df["multiclass_label"]

    # Categorical columns for encoding
    categorical_cols = ["protocol_type", "service", "flag"]
    categorical_cols = [c for c in categorical_cols if c in df.columns]

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Separate features and drop label columns
    feature_cols = [c for c in df.columns if c not in ["label", "binary_label", "multiclass_label"]]
    X = df[feature_cols].copy()

    # Label encoding for categoricals
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le

    # Ensure numeric
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    X = X.fillna(0)

    feature_names = list(X.columns)

    # Stratified split (80-20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=test_size, stratify=y, random_state=RANDOM_STATE
    )

    # Scale (fit only on train)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    result = {
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": feature_names,
        "label_encoder": label_encoder,
        "scaler": scaler,
        "encoders": encoders,
        "task": task,
        "class_names": list(label_encoder.classes_),
    }

    if save_artifacts:
        output_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, output_dir / "scaler.joblib")
        joblib.dump(label_encoder, output_dir / "label_encoder.joblib")
        joblib.dump(encoders, output_dir / "categorical_encoders.joblib")
        np.savez(
            output_dir / "processed_data.npz",
            X_train=X_train_scaled,
            X_test=X_test_scaled,
            y_train=y_train,
            y_test=y_test,
            feature_names=feature_names,
        )
        joblib.dump(result["class_names"], output_dir / "class_names.joblib")

    return result
