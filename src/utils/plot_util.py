from math import ceil

import matplotlib.pyplot as plt
import seaborn as sns


def generate_plots_eda(
    plot_type,
    data,
    x,
    y,
    x_label,
    y_label,
    plot_title,
    artefact_title=None,
    ax=None,
    plot_subplots=False,
):
    match plot_type:
        case "count_plot":
            sns.countplot(data=data, x=x)
        case "box_plot":
            sns.boxplot(data=data, x=x, ax=ax)
        case "hist_plot":
            sns.histplot(data=data, x=x, kde=True, bins=40)
        case "heatmap_plot":
            sns.heatmap(data=data)
        case "pair_plot":
            sns.pairplot(data=data)
        case "scatter_plot":
            sns.scatterplot(data=data, x=x, y=y)
        case _:
            sns.countplot(data=data, x=x)

    # tight layout to see the labels clearly spaced
    if plot_subplots == False:
        plt.tight_layout()
    
    format_plots(x_label, y_label, plot_title, ax=ax)

    if artefact_title:
        save_plots(plt, f"./artefacts/{artefact_title.replace(' ', '_')}.png")

    if plot_subplots == False:
        plt.close()


def format_plots(x_label: str, y_label: str, title, ax=None):
    if ax: 
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)
    else:     
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title)


def show_plots(plt):
    plt.show()


def save_plots(plt, path_to_save: str, dpi=200):
    plt.savefig(fname=path_to_save, dpi=dpi, bbox_inches="tight")


def generate_subplots(dataframe, artefact_title):
    features = dataframe.columns
    features_count = len(list(features))
    n_cols = 3
    n_rows = ceil(features_count / n_cols)
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(15, 4 * n_rows))

    # convert axes as 1D array
    axes = axes.flatten()

    for i, feature in enumerate(features):
        generate_plots_eda(
            "box_plot",
            data=dataframe,
            x=feature,
            y=None,
            x_label=feature,
            y_label="Values",
            plot_title=f"Boxplot of {feature}",
            ax=axes[i],
            plot_subplots=True,
        )

    # Clean up empty subplots if grid > feature count
    for j in range(features_count, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    save_plots(plt, f"./artefacts/{artefact_title.replace(' ', '_')}.png")

    plt.close()
