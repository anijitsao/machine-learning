## python-machine-learning
This porject demonstrates various concepts of Machine Learing. We have covered various Machine Learning algorithms like Regression and Classification.


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

