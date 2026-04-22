import joblib
import pandas as pd
import numpy as np
from src.data import generate_data, load_config
from src.uncertainty import calibrate_and_fit_mapie, get_prediction_sets

def run_business_simulation():
    config = load_config("configs/params.yaml")
    (X_train, y_train), (X_calib, y_calib), (X_test, y_test) = generate_data(config)
    model = joblib.load("data/model.joblib")
    mapie = calibrate_and_fit_mapie(model, X_calib, y_calib, config["conformal"]["alpha"])
    pred_sets = get_prediction_sets(mapie, X_test, config["conformal"]["alpha"])
    
    print("APPLICATION PROCESSING REPORT")
    print("App ID   Set Size   Decision")
    
    for i in range(10):
        set_size = np.sum(pred_sets[i])
        if set_size == 1:
            decision = "AUTO-APPROVE" if 0 in np.where(pred_sets[i])[0] else "AUTO-REJECT"
        else:
            decision = "MANUAL REVIEW REQUIRED"
        print(f"{i:<8} {set_size:<10} {decision}")

if __name__ == "__main__":
    run_business_simulation()
