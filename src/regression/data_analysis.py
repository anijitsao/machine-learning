import seaborn as sns

from src.utils.plot_labels_util import format_plots, show_plots


def analysis_data(df):
    # ploting counts for each class type
    sns.countplot(data=df, x="target")
    format_plots("Types of Wine", "Count", "Count of Various wine types")

    # plotting to check outliers
    sns.boxplot(data=df)
    format_plots(
        "Statistical values of Data", "Count", "Statistical description of data"
    )

