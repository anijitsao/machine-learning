from src.utils.plot_util import generate_plots_eda


def analysis_data(df):
    try:
        # ploting counts for each class type
        generate_plots_eda(
            "count_plot",
            df,
            "target",
            None,
            "Types of Wine",
            "Count",
            "Count of Various wine types",
            "count_plot_classification",
        )

        # plotting to check outliers
        generate_plots_eda(
            "box_plot",
            df,
            None,
            None,
            "Statistical values of Data",
            "Count",
            "Statistical description of data",
            "box_plot_classification",
        )

        # plotting heatmap
        heatmap_data = df.drop(columns=["target"]).corr()
        generate_plots_eda(
            "heatmap_plot",
            heatmap_data,
            None,
            None,
            "Attributes",
            "Attributes",
            "Correlation between Attributes",
            "correlation_classification"
        )
    except Exception as e:  # noqa: BLE001
        print("Error occurred while ploting", e)
