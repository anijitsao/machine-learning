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

        X_train, X_test, y_train, y_test = split_datasets(X, y, 0.6, "regression")

        base_model = get_base_model("regression")
        cross_validation = get_cross_validation("regression")
        params = get_randomized_search_params()

        # hyper parameter tuning
        search_model = get_randomized_search_model(
            base_model, params, cross_validation, scoring="neg_root_mean_squared_error"
        )
        search_model.fit(X_train, y_train)

        print("best parameters\n", search_model.best_params_)

        # generate the best model using best params
        final_model = get_final_model(search_model.best_params_, "regression")
        callbacks = get_callbacks()
        final_model.fit(
            X_train,
            y_train,
            eval_X=X_test,
            eval_y=y_test,
            eval_metric="rmse",
            callbacks=callbacks,
        )

        # predictions based on final model
        predictions = final_model.predict(X_test)

        # generate various scores
        regression_scores = generate_scores(y_test, predictions, "regression")
        print("Regression Report\n", regression_scores)

        # save the model or future use
        save_model(final_model, "regression.pkl")
    except Exception as e:  # noqa: BLE001
        print("Error occurred in Model Building", e)
