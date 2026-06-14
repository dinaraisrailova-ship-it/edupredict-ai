from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


def get_models():
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=42, multi_class="auto"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=10, random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42
        ),
        "XGBoost": XGBClassifier(
            n_estimators=200, learning_rate=0.1, max_depth=6,
            use_label_encoder=False, eval_metric="mlogloss",
            random_state=42, verbosity=0,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=200, learning_rate=0.1, max_depth=6,
            random_state=42, verbose=-1,
        ),
        "SVM": SVC(kernel="rbf", probability=True, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=7, n_jobs=-1),
        "Neural Network": MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            activation="relu",
            solver="adam",
            alpha=0.001,
            batch_size=64,
            learning_rate="adaptive",
            max_iter=300,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
        ),
    }


def get_ensemble(trained_models: dict):
    """Soft voting ensemble from top 3 tree models."""
    estimators = [
        ("XGBoost",       trained_models["XGBoost"]),
        ("LightGBM",      trained_models["LightGBM"]),
        ("Random Forest", trained_models["Random Forest"]),
    ]
    return VotingClassifier(estimators=estimators, voting="soft", n_jobs=-1)


MODEL_PARAMS = {
    "Logistic Regression": {
        "C": 1.0, "max_iter": 1000, "solver": "lbfgs", "multi_class": "auto"
    },
    "Random Forest": {
        "n_estimators": 200, "max_depth": 10, "min_samples_split": 2,
        "min_samples_leaf": 1, "max_features": "sqrt"
    },
    "Gradient Boosting": {
        "n_estimators": 200, "learning_rate": 0.1, "max_depth": 5,
        "subsample": 1.0, "min_samples_split": 2
    },
    "XGBoost": {
        "n_estimators": 200, "learning_rate": 0.1, "max_depth": 6,
        "subsample": 1.0, "colsample_bytree": 1.0, "gamma": 0
    },
    "LightGBM": {
        "n_estimators": 200, "learning_rate": 0.1, "max_depth": 6,
        "num_leaves": 31, "min_child_samples": 20
    },
    "SVM": {
        "C": 1.0, "kernel": "rbf", "gamma": "scale", "probability": True
    },
    "KNN": {
        "n_neighbors": 7, "weights": "uniform", "metric": "minkowski"
    },
    "Neural Network": {
        "hidden_layer_sizes": "(256, 128, 64)", "activation": "relu",
        "solver": "adam", "alpha": 0.001, "max_iter": 300
    },
    "Ensemble": {
        "voting": "soft", "estimators": "XGBoost + LightGBM + Random Forest"
    },
}
