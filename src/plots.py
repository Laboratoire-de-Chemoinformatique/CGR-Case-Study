"""Contains different colour palettes for data visualisation:"""

from pathlib import Path
from typing import List

import matplotlib
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

SYNTHWAVE_PALETTE = [
    "#003f5c",
    "#444e86",
    "#955196",
    "#dd5182",
    "#ff6e54",
    "#ffa600",
]

DIVERGENT_SYNTHWAVE_PALETTE = [
    "#003f5c",
    "#345871",
    "#587288",
    "#7a8d9e",
    "#9daab5",
    "#c1c7cd",
    "#e5e5e5",
    "#f0dac5",
    "#f7d0a5",
    "#fcc585",
    "#febb64",
    "#ffb040",
    "#ffa600",
]

SYNTHWAVE_CATEGORICAL_PALETTE = [
    "#003F5C",
    "#FFA600",
    "#D1495B",
    "#3E92CC",
    "#87BBA2",
    "#D4D2D5",
    "#D1BCE3",
    "#ff6e54",
]

SYNTHWAVE_SINGLE_HUE_PALETTE = [
    "#003f5c",
    "#2d5c7b",
    "#4f7b9c",
    "#709cbe",
    "#92bee1",
]


def initiate_matplotlib_settings(
    font_path: str = "/home/mball/Library/fonts/Roboto",
    style_path="/home/mball/Documents/projects/rcr-benchmarking/reports/figure_styles.mplstyle",
    font_name: str = None,
) -> None:
    """Adds fonts to matplotlib in case they aren't installed on the system. Initiates
    matplotlib style and font settings.

    :param font_path: Path to the font file.
    :type font_path: str
    :param style_path: Path to the matplotlib style file.
    :type style_path: str
    """
    font_dirs = [font_path]
    font_files = fm.findSystemFonts(fontpaths=font_dirs)

    for font_file in font_files:
        fm.fontManager.addfont(font_file)

    style_path = Path(style_path)
    plt.style.use(style_path)

    if font_name is not None:
        plt.rcParams["font.family"] = font_name


def get_continuous_cmap_from_palette(palette: List[str], n_colors: int):
    """Converts a discrete palette to a continuous colormap.

    :param palette: The palette (list of hexcode strings) to convert to a continuous colormap.
    :type palette: list[str]
    :param n_colors: Number of colours to interpolate between the discrete palette colours.
    :type n_colors: int
    :return: A continuous colormap.
    :rtype: LinearSegmentedColormap
    """
    cmap = LinearSegmentedColormap.from_list("custom", palette, N=n_colors)
    return cmap


def bolden_ax(
    ax,
    despine=False,
    spine_linewidth=2,
    tick_linewidth=2,
    label_fontsize=12,
    title_fontsize=14,
    legend_fontsize=12,
) -> None:
    """Helper function to bolden axis in a plot.

    :param ax: matplotlib axis object to bolden.
    :type ax: matplotlib.axis
    :param despine: controls wheter to despine the plot, defaults to False
    :type despine: bool, optional
    :param spine_linewidth: spine linewidth, defaults to 2
    :type spine_linewidth: int, optional
    :param tick_linewidth: tick linewidth, defaults to 2
    :type tick_linewidth: int, optional
    :param label_fontsize: label fontsize, defaults to 12
    :type label_fontsize: int, optional
    :param title_fontsize: title fontsize, defaults to 14
    :type title_fontsize: int, optional
    """
    if despine:
        sns.despine(ax=ax)

    # Make the axes bold:
    for axis in ["top", "bottom", "left", "right"]:
        if axis in ax.spines:
            ax.spines[axis].set_linewidth(spine_linewidth)

    if ax.xaxis is not None:
        ax.xaxis.set_tick_params(width=tick_linewidth)

    if ax.yaxis is not None:
        ax.yaxis.set_tick_params(width=tick_linewidth)

    # Finally get the labels and titles, make these bold:
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontsize(label_fontsize)
        label.set_fontweight("bold")

    for label in [ax.xaxis.label, ax.yaxis.label]:
        label.set_fontsize(label_fontsize)
        label.set_fontweight("bold")

    ax_title = ax.get_title()

    ax.set_title(ax_title, fontsize=title_fontsize, fontweight="bold")

    # Get the title and labels from the legend of the ax, and make them bold:
    legend = ax.get_legend()
    if legend is not None:
        legend_title = legend.get_title()
        legend_title.set_fontweight("bold")
        legend_title.set_fontsize(legend_fontsize)

        for label in legend.get_texts():
            label.set_fontweight("bold")
            label.set_fontsize(legend_fontsize)

        legend.get_frame().set_linewidth(spine_linewidth)


def extend_palette(n_colors: int, palette: List[str] = SYNTHWAVE_PALETTE):
    """Extends a palette of n colours by converting to a colormap and back into a list.

    Useful for when you want to extend a palette to a larger number of colours.

    :param n_colors: Number of colours in the palette.
    :type n_colors: int
    :param palette: The palette to use, defaults to SEQUENTIAL_PALETTE
    :type palette: list[str], optional
    :return: A sequential palette of n_colors.
    :rtype: list[str]
    """
    import matplotlib.colors as mcolors

    cmap = get_continuous_cmap_from_palette(palette, n_colors)

    return [mcolors.rgb2hex(cmap(i)) for i in range(n_colors)]


def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    - `lighten_color('g', 0.3)`
    - `lighten_color('#F034A3', 0.6)`
    - `lighten_color((.3,.55,.1), 0.5)`
    """
    import colorsys

    import matplotlib.colors as mc

    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


def set_boxplot_fill_styles(fill: bool, ax: matplotlib.axes.Axes) -> None:
    """Set the fill styles for the boxplots.

    If fill is false, colours the outline of the boxplot. If fill is true, fills the boxplot,
    darkening the edges.

    :param fill: Whether to fill the boxplots, or just colour the outline.
    :type fill: bool
    :param ax: The axis to set the fill styles on.
    :type ax: matplotlib.axes.Axes
    """
    box_patches = [
        patch
        for patch in ax.patches
        if type(patch) is matplotlib.patches.PathPatch
    ]
    if (
        len(box_patches) == 0
    ):  # in matplotlib older than 3.5, the boxes are stored in axs.artists
        box_patches = ax.artists

    num_patches = len(box_patches)
    lines_per_boxplot = len(ax.lines) // num_patches
    for i, patch in enumerate(box_patches):
        # Set the linecolor on the patch to the facecolor, and set the facecolor to None
        col = patch.get_facecolor()

        if not fill:
            patch.set_edgecolor(col)
            patch.set_facecolor("None")
        else:
            patch.set_edgecolor(lighten_color(col, 1.2))
            patch.set_linewidth(1)

        # Each box has associated Line2D objects (to make the whiskers, fliers, etc.)
        # Loop over them here, and use the same color as above
        for line in ax.lines[
            i * lines_per_boxplot : (i + 1) * lines_per_boxplot
        ]:
            if not fill:
                line.set_linewidth(1)
                line.set_mew(1)
                line.set_color(col)
                line.set_mfc("None")
                line.set_mec(col)
            else:
                line.set_linewidth(1)
                line.set_mew(1)
                line.set_color(lighten_color(col, 1.2))
                line.set_mfc("None")
                line.set_mec(lighten_color(col, 1.2))


def add_titles(
    fig: matplotlib.figure.Figure,
    title: str,
    subtitle: str,
    subtitle_y: float = 0.90,
) -> None:
    fig.suptitle(
        t=title,
        ha="left",
        x=0,
        y=1,
    )

    fig.text(
        x=0,
        y=subtitle_y,
        s=subtitle,
        fontdict={
            "ha": "left",
            "style": "italic",
            "weight": "regular",
        },
    )


def create_fig_legend(
    fig: matplotlib.figure.Figure,
    axs: matplotlib.axes.Axes,
    title: str = "Method",
    n_cols: int = 4,
    bbox_to_anchor: tuple = (0.7, 1.025),
    fill_legend: bool = True,
) -> None:
    handles, labels = axs[-1].get_legend_handles_labels()
    axs[-1].get_legend().remove()

    # Also fix the legend
    for legpatch in fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=bbox_to_anchor,
        ncol=n_cols,
        title=title,
    ).get_patches():
        col = legpatch.get_facecolor()
        if not fill_legend:
            legpatch.set_edgecolor(col)
            legpatch.set_facecolor("None")
        else:
            legpatch.set_edgecolor("None")
            legpatch.set_facecolor(col)
