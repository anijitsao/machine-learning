from .data_analysis import analysis_data
from .data_peparation import clean_data
from .model_building import build_model


def init_classification():
    # get the cleaned dataframe
    df = clean_data()
    analysis_data(df)
    build_model(df)
