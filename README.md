## python-machine-learning
This project demonstrates various concepts of Machine Learing (ML) pipeline. Various popular types of Machine Learning algorithms like Regression and Classification are covered. Each ML pipelines are started with structured `CSV` data loading. Then it passes through data cleaning and preparation, model training and building.

At last, all the models are saved in `PKL` files. And the models predict data when they hit by respective urls.

### Features
- All the ML models are trained with *structured* `CSV` data. For simplicity, built in datasets of `scikit-learn` is used.
- Data is **cleaned** and **processed** using `pandas`.
- **Exploratory Data Analysis (EDA)** is performed on the data. Necessary plots are generated using the famous plotting library `seaborn` and `matplotlib`.

<ul>
<li>Models are built using <code>scikit-learn</code> and popular <b>gradient boosting</b> library `LightGBM`</li>
<li><b>Hyperparameter tuning</b> of `LightGBM` is done using <i>Randomized Search CV</i>.</li>
<li><b>Cross validation</b> is done using <b>K-Fold</b> algorithms</li>
</ul>


- Models are saved into `PKL` format using `joblib`.

<ul>
<li>FastAPI is used to create the REST APIs.</li>
<li>Models predict data while hit by <i>respective</i> API endpoints.</li>
<li>Swagger Documentation is supported through FastAPI.</li>
</ul>


### Installation
`uv` is used as the package manager instead of python default package manager `pip`. While you are developing a project using `uv` as the package manager please use the following shell commands from a terminal or command prompt.

```shell
# Create/Navigate to the directory
mkdir python-machine-learning
cd python-maching-learning

# Initialize the project. It will create the pyproject.toml file and others
uv init

# Create a virtual environment and activate the same
uv venv
source .venv/bin/activate  # Linux
.venv\Scripts\activate # Windows

# Install various dependencies
uv add pandas # for a single package
uv add seaborn xgboost joblib # for multiple packages
```

When you have cloned the repository from GitHub and want to install those dependencies (in that case you already have the `pyproject.toml` file) please use the following commands.

```shell
git clone https://github.com/anijitsao/python-machine-learning.git

# Navigate into the directory
cd python-machine-learning

# Activate the virtual environment
source .venv/bin/activate  # Linux
.venv\Scripts\activate # Windows

# Install all those dependencies
uv sync
```

#### Run the Machine Learning (ML) Pipeline
To run the Machine Learning pipeline please use the following commands from a terminal.

```shell
# Navigate to the directory
cd python-machine-learning

# Run the pipeline
python main.py
```

#### Run the FastAPI Application
FastAPI is used to create the necessary REST APIs for this application. To run the FastAPI application please use the following commands from a terminal. 

> [!IMPORTANT]

> Please make sure you have run the Machine Learning pipeline before running the FastAPI application.

```shell
# Navigate to the directory
cd python-machine-learning

# Run the FastAPI server
uv run fastapi dev app.py # development
uv run fastapi run app.py # production
```

Now please go to [http://localhost:8000/docs](http://localhost:8000/docs) and play with the APIs