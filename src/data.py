import yaml
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from loguru import logger

def load_config(path="configs/params.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def generate_data(config):
    logger.info("Generating synthetic data")
    X, y = make_classification(
        n_samples=config["data"]["n_samples"],
        n_features=config["data"]["n_features"],
        class_sep=config["data"]["class_sep"],
        weights=config["data"]["weights"],
        random_state=config["model"]["random_state"]
    )
    
    # 3-way split
    # train (60%), calib (20%), test (20%)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=config["data"]["test_size"], random_state=config["model"]["random_state"]
    )
    
    calib_size_adj = config["data"]["calib_size"] / (1 - config["data"]["test_size"])
    X_train, X_calib, y_train, y_calib = train_test_split(
        X_temp, y_temp, test_size=calib_size_adj, random_state=config["model"]["random_state"]
    )
    
    # Save processed data
    Path("data").mkdir(exist_ok=True)
    df_train = pd.DataFrame(X_train).assign(target=y_train)
    df_calib = pd.DataFrame(X_calib).assign(target=y_calib)
    df_test = pd.DataFrame(X_test).assign(target=y_test)
    
    pd.concat([df_train, df_calib, df_test]).to_parquet("data/processed.parquet")
    
    # Check leakage
    min_len = min(len(y_train), len(y_test))
    corr = np.corrcoef(y_train[:min_len], y_test[:min_len])[0, 1]
    if corr > 0.01:
        logger.warning(f"Potential leakage detected: correlation {corr:.4f}")
    
    return (X_train, y_train), (X_calib, y_calib), (X_test, y_test)
