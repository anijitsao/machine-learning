##
# This file plots various types of plots
# It is required for the EDA purpose
##
# import pandas as pd
from sklearn.datasets import load_diabetes


def clean_data():
    # loading built in datasets from sklearn
    data = load_diabetes(as_frame=True)

    # get it as pandas dataframe
    df = data["frame"]
    # print(df.head())

    print(f"Total number of null values present\n {df.isnull().sum()}")

    # as there is no missing values we return the dataframe
    return df


# initialize
# clean_data()
