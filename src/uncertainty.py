import numpy as np
from mapie.classification import _MapieClassifier as MapieClassifier
from sklearn.calibration import CalibratedClassifierCV
from loguru import logger

def calibrate_and_fit_mapie(model, X_calib, y_calib, alpha):
    logger.info("Calibrating model and fitting Mapie")
    
    # Calibrate base classifier
    base_clf = model.named_steps["classifier"]
    calibrated_clf = CalibratedClassifierCV(base_clf, method='isotonic', cv=5)
    
    # Update pipeline
    model.set_params(classifier=calibrated_clf)
    model.fit(X_calib, y_calib) 

    # Use the internal class directly
    mapie = MapieClassifier(estimator=model, cv="prefit")
    mapie.fit(X_calib, y_calib) 
    
    return mapie

def get_prediction_sets(mapie, X_test, alpha):
    _, y_pis = mapie.predict(X_test, alpha=[alpha])
    return y_pis[:, :, 0]
