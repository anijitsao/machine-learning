from .data_analysis import analysis_data
from .data_peparation import clean_data


def init_classification():
    # get the cleaned dataframe
    df = clean_data()
    analysis_data(df)
