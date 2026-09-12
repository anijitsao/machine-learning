from src.utils import (
    generate_scores,
    get_base_model,
    get_callbacks,
    get_cross_validation,
    get_final_model,
    get_randomized_search_model,
    get_randomized_search_params,
    save_model,
    split_datasets,
)


def build_model(df):
    try:
        X = df.drop(columns="target")
        y = df["target"]

        X_train, X_temp, y_train, y_temp = split_datasets(X, y, 0.6)
        X_test, X_val, y_test, y_val = split_datasets(X_temp, y_temp, 0.2)
        base_model = get_base_model()
        cross_validation = get_cross_validation()
        params = get_randomized_search_params()

        # hyper parameter tuning
        search_model = get_randomized_search_model(base_model, params, cross_validation)
        search_model.fit(X_train, y_train)

        print("best parameters\n", search_model.best_params_)

        # generate the best model using best params
        final_model = get_final_model(search_model.best_params_)
        callbacks = get_callbacks()
        final_model.fit(
            X_train,
            y_train,
            eval_X=X_val,
            eval_y=y_val,
            eval_metric="multi_logloss",
            callbacks=callbacks,
        )

        # predictions based on final model
        predictions = final_model.predict(X_test)

        # generate various scores
        classification_scores = generate_scores(y_test, predictions)
        print("Classification Report\n", classification_scores)

        # save the model or future use
        save_model(final_model)
    except Exception as e:  # noqa: BLE001
        print("Error occurred in Model Building", e)
