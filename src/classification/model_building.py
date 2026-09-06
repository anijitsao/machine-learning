from src.utils import (
    generate_scores,
    get_base_model,
    get_callbacks,
    get_cross_validation,
    get_final_model,
    get_randomized_search_model,
    get_randomized_search_params,
    split_datasets,
)


def build_model(df):
    X = df.drop(columns="target")
    y = df["target"]

    X_train, X_test, y_train, y_test = split_datasets(
        X, y, train_size=0.6, random_state=42
    )

    base_model = get_base_model()
    cross_validation = get_cross_validation()
    params = get_randomized_search_params()
    search_model = get_randomized_search_model(base_model, params, cross_validation)
    search_model.fit(X_train, y_train)

    print("best parameters\n", search_model.best_params_)

    final_model = get_final_model(search_model.best_params_)
    callbacks = get_callbacks()
    final_model.fit(
        X_train,
        y_train,
        eval_X=X_test,
        eval_y=y_test,
        eval_metric="multi_logloss",
        callbacks=callbacks,
    )

    predictions = final_model.predict(X_test)
    classification_scores = generate_scores(y_test, predictions)
    print("Classification Report\n", classification_scores)
