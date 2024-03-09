from typing import List, Tuple
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class Plots:
    def __init__(self, data: pd.DataFrame, cat_cols: List[str] = None, cont_cols: List[str] = None):
        self.data = data
        self._cat_cols = cat_cols
        self.cont_cols = cont_cols

    @property
    def cat_cols(self):
        return self._cat_cols

    @cat_cols.setter
    def cat_cols(self, val_list: List[str]):
        self._cat_cols = val_list

    def _plot_single(self, ax, col):
        sns.countplot(ax=ax, x=col, data=self.data)
        ax.set_title(f"{col} counts plot")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

        # ax.set_xticklabels(ax., size=12, rotation=45)
        # ax.xlabel(col, size=12)
        # ax.yticks(size=12)

        for p in ax.patches:
            ax.annotate(
                p.get_height(),
                xy=(p.get_x() + p.get_width() / 2, p.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center',
                va='bottom'
            )

    def plot_all_cats(self, nrows: int = 1, ncols: int = 1,
                      figsize: Tuple = (10, 8), title: str = "Categorical columns count plots"):
        sns.set_theme()
        if self.cat_cols is None:
            return "No categorical column provided"

        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        fig.suptitle(title)
        for name, ax in zip(self._cat_cols, axes.flatten() if len(axes) >= 2 else axes):
            self._plot_single(ax=ax, col=name)
        plt.tight_layout()
