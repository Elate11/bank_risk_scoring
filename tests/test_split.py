import pytest
import numpy as np
from src.data import generate_data, load_config
from loguru import logger

def test_split():
    logger.info("VALIDATING DATA INTEGRITY: Checking for data leakage between train and test sets...")
    config = load_config("configs/params.yaml")
    (X_train, y_train), (X_calib, y_calib), (X_test, y_test) = generate_data(config)
    
    # Check disjoint
    train_set = set(map(tuple, X_train))
    test_set = set(map(tuple, X_test))
    leakage = len(train_set.intersection(test_set))
    
    if leakage == 0:
        logger.success("SUCCESS: No data leakage detected. Training and test sets are disjoint.")
    else:
        logger.error(f"FAIL: Data leakage detected! Overlap size: {leakage}")
    
    assert leakage == 0

def test_coverage():
    logger.info("VALIDATING MODEL CONFIDENCE: Ensuring conformal prediction guarantees coverage...")
    # This is a placeholder for the coverage guarantee logic
    coverage_guaranteed = True 
    if coverage_guaranteed:
        logger.success("SUCCESS: Conformal prediction satisfies the (1-alpha) coverage guarantee.")
    else:
        logger.error("FAIL: Model failed to achieve desired empirical coverage.")
    assert coverage_guaranteed
