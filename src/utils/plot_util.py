import matplotlib.pyplot as plt
import seaborn as sns


def generate_plots_eda(
    plot_type, data, x, y, x_label, y_label, plot_title, artefact_title
):
    match plot_type:
        case "count_plot":
            sns.countplot(data=data, x=x)
        case "box_plot":
            sns.boxplot(data=data, x=x)
        case "hist_plot":
            sns.histplot(data=data, x=x, kde=True)
        case "heatmap_plot":
            sns.heatmap(data=data)
        case "pair_plot":
            sns.pairplot(data=data, x=x, y=y)
        case _:
            sns.countplot(data=data, x=x)

    # tight layout to see the labels clearly spaced
    plt.tight_layout()
    format_plots(x_label, y_label, plot_title)
    save_plots(plt, f"./artefacts/{artefact_title.replace(' ', '_')}.png")
    plt.close()

def format_plots(x_label: str, y_label: str, title):
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)


def show_plots(plt):
    plt.show()


def save_plots(plt, path_to_save: str, dpi=200):
    plt.savefig(fname=path_to_save, dpi=dpi, bbox_inches="tight")
