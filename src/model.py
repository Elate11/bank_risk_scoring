import joblib
from lightgbm import LGBMClassifier
from loguru import logger
from sklearn.pipeline import Pipeline
from src.features import get_preprocessor

def train_model(config, X_train, y_train):
    logger.info("Training LightGBM model")
    model = Pipeline([
        ("preprocessor", get_preprocessor()),
        ("classifier", LGBMClassifier(
            learning_rate=config["model"]["learning_rate"],
            max_depth=config["model"]["max_depth"],
            n_estimators=config["model"]["n_estimators"],
            random_state=config["model"]["random_state"],
            verbose=-1
        ))
    ])
    
    model.fit(X_train, y_train)
    joblib.dump(model, "data/model.joblib")
    return model
