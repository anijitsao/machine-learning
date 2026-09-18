# from random import randint, uniform
from pathlib import Path

import joblib
from lightgbm import LGBMClassifier, LGBMRegressor, early_stopping
from scipy.stats import loguniform, randint, uniform
from sklearn.metrics import classification_report, r2_score, root_mean_squared_error
from sklearn.model_selection import (
    KFold,
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)


def get_base_model(task_type="classification", random_state=42):
    base_model = None
    if task_type == "classification":
        base_model = LGBMClassifier(random_state=random_state, verbosity=-1)
    else:
        base_model = LGBMRegressor(random_state=42, verbosity=-1)
    return base_model


def get_cross_validation(task_type="classification", n_splits=3, random_state=42):
    cross_validation = None
    if task_type == "classification":
        cross_validation = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=random_state
        )
    else:
        cross_validation = KFold(
            n_splits=n_splits, shuffle=True, random_state=random_state
        )

    return cross_validation


def get_randomized_search_params(
    n_estimators=(1000, 2000),
    learning_rate=(0.005, 0.1),
    max_depth=(2, 6),
    num_leaves=(4, 16),
    min_child_samples=(15, 60),
    reg_alpha=(0.1, 10),
    reg_lambda=(0.1, 10),
    subsample=(0.5, 0.9),
    colsample_bytree=(0.4, 0.5),
):
    params = {
        "n_estimators": randint(*n_estimators),
        "learning_rate": loguniform(*learning_rate),
        "max_depth": randint(*max_depth),
        # only Light GBM uses following
        "num_leaves": randint(*num_leaves),  # for XG Boost use min_child_weight
        "min_child_samples": randint(*min_child_samples),  # for XG Boost use gamma
        "reg_alpha": loguniform(*reg_alpha),
        "reg_lambda": loguniform(*reg_lambda),
        # Stochastic regularisation (crucial for small datasets)
        "subsample": uniform(*subsample),
        "colsample_bytree": uniform(*colsample_bytree),
    }

    return params


def split_datasets(X, y, train_size, task_type="classification", random_state=42):
    X_train = None
    X_test = None
    y_train = None
    y_test = None
    if task_type == "classification":
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=train_size, random_state=random_state, stratify=y
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=train_size, random_state=random_state
        )
    # print()
    return X_train, X_test, y_train, y_test


def get_randomized_search_model(
    estimator,
    params,
    cross_validation,
    iteration=3,
    scoring="f1_macro",
    random_state=42,
):
    search_model = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=params,
        cv=cross_validation,
        n_iter=iteration,
        scoring=scoring,
        random_state=random_state,
    )

    return search_model


def get_final_model(best_params, task_type="classification", random_state=42):
    final_model = None
    if task_type == "classification":
        final_model = LGBMClassifier(**best_params, random_state=random_state)
    else:
        final_model = LGBMRegressor(**best_params, random_state=random_state)
    return final_model


def get_callbacks(stopping_rounds=30):
    return [early_stopping(stopping_rounds)]


def generate_scores(
    true_data,
    predictions,
    task_type="classification",
):
    scores = None
    if task_type == "classification":
        # use output_dict=True if you want to print as a dictionary
        scores = classification_report(y_true=true_data, y_pred=predictions, digits=2)
    else:
        rmse_score = root_mean_squared_error(y_true=true_data, y_pred=predictions)
        r2_score_value = r2_score(y_true=true_data, y_pred=predictions)
        scores = {"rmse": round(rmse_score, 2), "r2": round(r2_score_value, 2)}
    return scores


def save_model(
    model,
    file_name="classification.pkl",
    path="./models",
):
    folder_path = Path(path)
    folder_path.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, f"{path}/{file_name}")
    print("Model saved successfully")
