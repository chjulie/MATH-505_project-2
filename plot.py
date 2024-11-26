import numpy as np


def setColNames(
    axes,
    cols=["MNIST RBF", "Poly Decay", "Exp Decay"],
    rows=["R = 5", "R = 10", "R = 20"],
):
    """
    for the grid of plots set column and row names
    """
    if type(axes[0]) != np.ndarray:
        for i in range(len(axes)):
            axes[i].set(xlabel=cols[i], ylabel=rows[0])
    else:
        for ax, col in zip(axes[0], cols):
            ax.set_title(col)
        for ax, row in zip(axes[:, 0], rows):
            ax.set_ylabel(row, rotation=0, size="large")


def setIndividualTitles(axes, titles=["MNIST RBF", "Poly Decay", "Exp Decay"]):
    """
    Set individual titles for subplots
    """
    if type(axes[0]) != np.ndarray:
        axes = axes.reshape(1, len(axes))
    for i in range(axes.shape[1]):
        axes[0, i].set_title(titles[i])
