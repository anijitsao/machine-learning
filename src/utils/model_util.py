# from random import randint, uniform
from pathlib import Path

import joblib
from lightgbm import LGBMClassifier, LGBMRegressor, early_stopping
from scipy.stats import randint, uniform
from sklearn.metrics import classification_report, root_mean_squared_error
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


def get_randomized_search_params():
    params = {
        "n_estimators": randint(100, 1000),
        "learning_rate": uniform(0.01, 0.1),
        "max_depth": randint(3, 5),
        # only Light GBM uses following
        "num_leaves": randint(15, 100),  # for XG Boost use min_child_weight
        "min_child_samples": randint(5, 50),  # for XG Boost use gamma
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
    random_state=42,
    scoring="f1_macro",
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


def get_callbacks():
    return [early_stopping(30)]


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
        scores = root_mean_squared_error(y_true=true_data, y_pred=predictions)
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
