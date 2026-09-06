# from random import randint, uniform
from lightgbm import LGBMClassifier, LGBMRegressor, early_stopping
from scipy.stats import randint, uniform
from sklearn.metrics import classification_report
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)


def get_base_model(objective="multiclass", random_state=42):
    base_model = LGBMClassifier(
        objective=objective, random_state=random_state, verbosity=-1
    )
    return base_model


def get_cross_validation(n_splits=3, random_state=42):
    cross_validation = StratifiedKFold(
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


def split_datasets(X, y, train_size, random_state=42):
    splitted_datasets = train_test_split(
        X, y, train_size=train_size, random_state=random_state
    )
    # print()
    return splitted_datasets


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


def get_final_model(best_params, objective="multiclass", random_state=42):
    final_model = LGBMClassifier(
        **best_params, objective=objective, random_state=random_state
    )
    return final_model


def get_callbacks():
    return [early_stopping(30)]


def generate_scores(
    true_data,
    predictions,
    objective="multiclass",
):
    # use output_dict=True if you want to print as a dictionary
    scores = classification_report(y_true=true_data, y_pred=predictions, digits=2)
    return scores
