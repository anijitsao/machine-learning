from src.utils.plot_util import generate_plots_eda


def analysis_data(df):
    try:
        # ploting counts for each class type
        generate_plots_eda(
            "hist_plot",
            df,
            "target",
            None,
            "Diabetes probability",
            "Count",
            "Diabetes Progression Score",
            "hist_plot_regression",
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
            "box_plot_regression",
        )

        # plotting pair plot
        generate_plots_eda(
            "scatter_plot",
            df,
            'target',
            'age',
            "Diabetes Score",
            "Age",
            "Relation of Daibetes score with Age",
            "scatter_plot_regression",
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
            "correlation_regression",
        )
    except Exception as e:  # noqa: BLE001
        print("Error occurred while ploting", e)
