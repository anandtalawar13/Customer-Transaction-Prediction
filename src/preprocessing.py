import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(path):
    """Load dataset and return a pandas DataFrame."""
    df = pd.read_csv(path)
    return df


def split_features_target(df, target_col="target"):
    """Split dataframe into X and y."""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def scale_features(X_train, X_test):
    """Scale features using StandardScaler."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def preprocess_dataset(path, test_size=0.2, random_state=42):
    """Complete preprocessing pipeline."""
    df = load_data(path)
    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler
