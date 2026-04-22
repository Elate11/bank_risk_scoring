from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def get_preprocessor():
    """Returns minimal preprocessing pipeline."""
    return Pipeline([
        ("scaler", StandardScaler())
    ])
