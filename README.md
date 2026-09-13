## python-machine-learning
This project demonstrates various concepts of Machine Learing (ML). Various popular types of Machine Learning algorithms like Regression and Classification are covered. Each ML pipelines are stated with structured CSV data loading. Then it passes through data cleaning and preparation, model training and building.

At last all the models are saved in `PKL` files. All the models predicts when they hit by respective urls.

### Features
- All the ML models are trained with structured CSV data. For simplcity built in datasets of `scikit-learn` is used.
- Data is cleaned and processed using `pandas`.
- **Exploratory Data Analysis (EDA)** is performed on the data. Necessary plots are generated using the famous plotting library `seaborn` and `matplotlib`.

- Models are built using `scikit-learn` and popular gradient boosting library `LightGBM`
- Hyperparameter tuning of `LightGBM` is done using *Randomized Search CV*.

- Models are saved into `PKL` format using `joblib`.

- `FastAPI` is used to create the REST APIs
- Models predict data while upon respective API endpoints.


### Installation
`uv` is used as the package manager instead of python default package manager `pip`. While you are developing a project using `uv` as the package manager please use the following shell commands from a terminal or command prompt.

```shell
# Initialize the project. It will create the pyproject.toml file and others
uv init

# Create a virtual environment and activate the same
uv venv
source .venv/bin/activate 

# Install various dependencies
uv add pandas
uv add seaborn xgboost joblib

```

When you have cloned the repository from GitHub and want to install those dependencies (in that case you already have the `pyproject.toml` file) please use the following commands.

```shell
git clone https://github.com/anijeet_sao/python-machine-learning

# Navigate into the directory
mkdir python-machine-learning

# Install all those dependencies
uv sync
```

