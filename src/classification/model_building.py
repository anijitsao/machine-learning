from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split


def build_model(df):
    X = df.drop(columns="target")
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.6, random_state=42
    )

    model = LGBMClassifier(
        learning_rate=0.1,
        num_leaves=31,
        n_estimators=100,
        max_depth=5
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    precision = precision_score(y_test, predictions, average="weighted")
    recall = recall_score(y_test, predictions, average="weighted")
    f1 = f1_score(y_test, predictions, average="weighted")


    print(f"Precision score: {precision}")
    print(f"Recall score: {recall}")
    print(f"F1 score: {f1}")


