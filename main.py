from src.classification import init_classification
from src.regression import init_regression


def main():
    print("Hello from python-machine-learning!")

    # perform classification
    init_classification()

    # perform regression
    init_regression()


if __name__ == "__main__":
    main()


"""
TO DO:
    - Regression
    - Multiple Box Plots for EDA
    - LightGBM Model Building, train, scores etc.
    - Save models using joblib 
    - FastAPI routes to serve Predictions    

"""