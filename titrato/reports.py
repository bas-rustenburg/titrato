import string
from copy import deepcopy
from shutil import copyfile
from typing import List, Tuple, Dict, Optional
import warnings
import re
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
from adjustText import adjust_text
from matplotlib import pyplot as plt
from tqdm.autonotebook import tqdm, trange
from matplotlib.ticker import MaxNLocator, MultipleLocator
from matplotlib.font_manager import FontProperties

import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from titrato import closest_pka, hungarian_pka, align_pka
from titrato import fit_titration_curves_3d
from titrato.stats import (
    absolute_loss,
    squared_loss,
    array_mae,
    array_rmse,
    wrap_pearsonr,
    bootstrapped_func,
    array_median_error,
)
from .sampl import (
    TitrationCurveType,
    SAMPL6DataProvider,
    bootstrap_rmse_r,
    bootstrap_pKa_dataframe,
    HaspKaType,
    TypeIPrediction,
)
from .stats import (
    area_between_curves,
    area_curve_vectorized,
    rmsd_curve_vectorized,
    BootstrapDistribution,
    area_between_curves_ne,
)
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
import logging
from typing import List
from uncertainties import ufloat
import networkx as nx
from networkx.drawing.nx_pydot import pydot_layout
from collections import deque

log = logging.getLogger()

# Default styling
sns.set_style("ticks")
glob_font = {"size": 8}
matplotlib.rc("font", **glob_font)
matplotlib.rc("lines", **{"markersize": 4})
# Default colors per charge
charge_colors = {
    -4: "#470911",
    -3: "#b2182b",
    -2: "#d6604d",
    -1: "#f4a582",
    0: "#333333",
    1: "#92c5de",
    2: "#4393c3",
    3: "#2166ac",
    4: "#0d2844",
}

import os


def to_str(num: ufloat):
    """Formats ufloat with one precision digit on the uncertainty and latex syntax"""
    return "{:.1uL}".format(num)


class TexBlock:
    """Basic class for latex syntax block, to be added to report."""

    tex_src = ""

    def __init__(self, **kwargs):
        self._variables = kwargs
        return

    def render_source(self):
        """Fill in all variable fields and return complete latex source."""
        return self.tex_src.format(**self._variables)


class ReportHeader(TexBlock):
    """Represents the preamble and document start."""

    tex_src = (
        "\\documentclass[9pt]{{standalone}}\n"
        "\\renewcommand{{\\familydefault}}{{\\sfdefault}}\n"
        "\\usepackage[utf8]{{inputenc}}\n"
        "\\usepackage{{graphicx}}\n"
        "\n\\begin{{document}}\n"
    )

    # syntax for adding a new variable
    tex_var = "\\newcommand{{\\{}}}{{{}}}\n"

    def __init__(self, mol_id, method_names, img_ext="pdf"):
        """Initialize the header of the file by setting all appropriate variables"""
        variables = dict()
        variables["molid"] = mol_id
        variables["imgext"] = img_ext
        self.ids = list()
        # need to assign ascii name to method for use as tex variable
        for method, name in enumerate(method_names):
            id = string.ascii_lowercase[method]
            variables[f"method{id}"] = name
            # Keep track of ids defined in header
            self.ids.append(id)

        self._variables = variables

    def render_source(self):

        src = self.tex_src.format()
        for name, value in self._variables.items():
            src += self.tex_var.format(name, value)
        return src


class ReportFooter(TexBlock):
    tex_src = "\\end{{document}}\n"


class OverviewRow(TexBlock):
    """Latex syntax provider for the overview section."""

    tex_src = (
        "\\section{{\\molid}}"
        "\n\\noindent \n"
        "\\begin{{minipage}}[s]{{0.35\\textwidth}}\\centering\n"
        "\\includegraphics[width=\\textwidth]{{Reports/\\molid-molecule.\\imgext}}\n"
        "\\end{{minipage}}\n"
        "\\begin{{minipage}}[s]{{0.35\\textwidth}}\n"
        "\\includegraphics[width=\\textwidth]{{Reports/overview-virtual-titration-\\molid.\\imgext}}\n"
        "\\end{{minipage}}\n"
        "\\begin{{minipage}}[s]{{0.23\\textwidth}}\n"
        "\\includegraphics[width=\\textwidth]{{Reports/overview-legend-\\molid.\\imgext}}\n"
        "\\end{{minipage}}\n"
    )


class MethodResultRow(TexBlock):
    """A row of figures for a single method"""

    tex_src = (
        "\n\\begin{{minipage}}[s]{{\\textwidth}}\\centering\n"
        "{{\\textbf \\method{id}}}\n"
        "\\end{{minipage}}\n"
        "\n\\noindent\n"
        "\\begin{{minipage}}[s]{{0.33\\textwidth}}\\centering\n"
        "\\includegraphics[width=\\textwidth]{{Reports/\\method{id}-virtual-titration-\\molid.\\imgext}}\n"
        "\\end{{minipage}}\n"
        "\\begin{{minipage}}[s]{{0.33\\textwidth}}\n"
        "\\includegraphics[\\textwidth]{{Reports/\\method{id}-free-energy-\\molid.\\imgext}}\n"
        "\\end{{minipage}}\n"
        "\\begin{{minipage}}[s]{{0.33\\textwidth}}\n"
        "\\includegraphics[\\textwidth]{{Reports/\\method{id}-populations-\\molid.\\imgext}}\n"
        "\\end{{minipage}}\n"
    )

    def __init__(self, id):
        """A row of figures for a single method.

        Parameters
        id - the 1-letter identifier for the method (a-z).
        """
        self._variables = dict(id=id)


class SAMPL6ReportGenerator:
    """This class provides an interface for generating analysis plots between experiment, and a prediction for a single molecule."""

    # Assume pH values are spaced apart by 0.1 for any integration purposes.
    _dpH = 0.1

    # Plotting defaults

    _figprops = {
        "dpi": 150,
        "figsize": (2.0, 2.0),  # 3 figures fitting between 3 cm margins on letter paper
        "line_styles": ["-", "--", "-.", ":"],
        "line_widths": [0.75, 1.25, 1.25, 1.25],
        "colors_per_charge": charge_colors,
        # Use consistent colors for each method
        "extra_colors": sns.color_palette("dark"),
    }

    # Default number of bootstrap samples used to estimate titration curve confidence intervals
    num_bootstrap_curves = 10000

    def __init__(
        self,
        mol_id: str,
        exp_provider: SAMPL6DataProvider,
        data_providers: List[SAMPL6DataProvider],
        mol_img_loc: str,
    ) -> None:
        """Instantiate the analysis from the identifier of the molecule, and providers of the data.

        Parameters
        ----------
        mol_id - molecule associated with this report
        exp_provider - provider for the experimental data source
        prediction_provides - list of providers for all the predictions
        mol_png_loc - location where an image of the molecule can be found
        """

        self._exp_provider = exp_provider
        self._prediction_providers = data_providers
        self._figures: Dict[str, Dict[str, matplotlib.figure.Figure]] = {
            pred.method_desc: dict() for pred in data_providers
        }
        # Add dict for overview figures
        self._figures["overview"] = dict()
        self._figures[exp_provider.method_desc] = dict()
        # Data tables by description, and latex format
        self._tables: Dict[str, str] = dict()
        # Latex Report document
        self._tex_source = ""
        self._num_predictions = len(data_providers)
        self._mol_id = mol_id
        self._mol_img = mol_img_loc

        return

    def _plot_charge_legend(self):
        """Generate a legend for all charges."""
        fig, ax = self._newfig()
        for charge in range(-4, 5):
            color = self._figprops["colors_per_charge"][charge]
            ax.plot([0, 1], [0, 1], color=color, label=f"{charge:+d}")
        # Separate legend figure
        figlegend, axlegend = plt.subplots(
            1, 1, figsize=[8, 0.5], dpi=self._figprops["dpi"]
        )
        handles, labels = ax.get_legend_handles_labels()
        # handles = np.concatenate((handles[::2],handles[1::2]),axis=0)
        # labels = np.concatenate((labels[::2],labels[1::2]),axis=0)

        leg = figlegend.legend(handles, labels, loc="center", ncol=9)
        axlegend.get_xaxis().set_visible(False)
        axlegend.get_yaxis().set_visible(False)
        for spine in ["top", "left", "bottom", "right"]:
            axlegend.spines[spine].set_visible(False)

        self._figures["overview"]["charge-legend"] = figlegend
        plt.close(fig)

    def make_all_plots(self):
        """Make all available plots for each prediction and the experiment.."""

        # self._plot_virtual_titration_overview()
        self._plot_charge_legend()
        # overview plot
        self._plot_virtual_titration_overview()

        # Experiment gets its own plots

        # Virtual titration plot
        figtype = "virtual-titration"
        newfig = self.plot_virtual_titration(self._exp_provider)
        self._figures["Experiment"][figtype] = newfig

        # Free enery values
        figtype = "free-energy"
        newfig = self.plot_predicted_free_energy(self._exp_provider)
        self._figures["Experiment"][figtype] = newfig
        # Populations
        figtype = "populations"
        newfig = self.plot_predicted_population(self._exp_provider)
        self._figures["Experiment"][figtype] = newfig

        # Each method gets its own plots
        for p, pred_loader in enumerate(self._prediction_providers):
            desc = pred_loader.method_desc
            # Virtual titration plot
            figtype = "virtual-titration"
            newfig = self.plot_virtual_titration(
                self._exp_provider, pred_loader=pred_loader, index=p
            )
            self._figures[desc][figtype] = newfig
            # Free enery values
            figtype = "free-energy"
            newfig = self.plot_predicted_free_energy(pred_loader)
            self._figures[desc][figtype] = newfig
            # Populations
            figtype = "populations"
            newfig = self.plot_predicted_population(pred_loader)
            self._figures[desc][figtype] = newfig

    def _plot_virtual_titration_overview(self):
        """Plot an overview of all methods using the virtual charge titration.

        Also stores a legend with color codes for each method, that can be used with other overview figures.
        """

        # TODO fill in the new structure for experimental plots

        # Overview charge titration
        titration_fig_ax = self._newfig()

        for idx, pred in enumerate(self._prediction_providers, start=0):
            desc = pred.method_desc
            if pred.can_bootstrap:
                exp_data, exp_curves, exp_bootstrap_data = (
                    self._load_experiment_with_bootstrap()
                )
                pred_data, bootstrap_data = pred.bootstrap(
                    self._mol_id, self.num_bootstrap_curves
                )
                pred_data.align_mean_charge(exp_data, area_between_curves, self._dpH)
                # Align all to experiment curve (note this is a joint bootstrap of experiment and prediction)
                curves = list()
                for dat, exp_dat in zip(bootstrap_data, exp_bootstrap_data):
                    dat.align_mean_charge(exp_dat, area_between_curves, self._dpH)
                    curves.append(deepcopy(dat.mean_charge))
                curves = np.asarray(curves)
                self._add_virtual_titration_bootstrap_sd(
                    titration_fig_ax, desc, pred_data, curves, idx, linestyle="-"
                )
                # experiment plotted as dashed line
                self._add_virtual_titration_bootstrap_sd(
                    titration_fig_ax,
                    f"{desc}-exp",
                    exp_data,
                    exp_curves,
                    idx,
                    linestyle="--",
                    alpha=0.5,
                )
            else:
                exp_data = self._exp_provider.load(self._mol_id)
                pred_data = pred.load(self._mol_id)
                pred_data.align_mean_charge(exp_data, area_between_curves, self._dpH)
                curve = pred_data.mean_charge
                self._add_virtual_titration_bootstrap_sd(
                    titration_fig_ax, desc, pred_data, np.asarray([curve]), idx
                )

        # Unpack tuple.
        fig, ax = titration_fig_ax
        ax.set_title(f"{self._mol_id}", fontsize=9)
        # Integer labels for y axis
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        # No labels on y axis, but indicate the integer values with ticks
        # labels = [item.get_text() for item in ax.get_yticklabels()]
        # empty_string_labels = [""] * len(labels)
        # ax.set_yticklabels(empty_string_labels)
        ax.set_ylabel(r"$Q_\mathsf{avg}$")
        ax.set_xlabel("pH")
        # x-tick every 2 pH units
        ax.set_xticks(np.arange(2.0, 14.0, 2.0))
        # remove top and right spines
        sns.despine()
        # fit everything within bounds
        fig.tight_layout()

        # Separate legend figure
        figlegend, axlegend = self._newfig()
        leg = figlegend.legend(*ax.get_legend_handles_labels(), loc="center")
        axlegend.get_xaxis().set_visible(False)
        axlegend.get_yaxis().set_visible(False)
        for spine in ["top", "left", "bottom", "right"]:
            axlegend.spines[spine].set_visible(False)

        self._figures["overview"]["virtual-titration"] = fig
        self._figures["overview"]["legend"] = figlegend

    def _load_experiment_with_bootstrap(self):
        # All methods tested against the same experimental values.
        exp_data = self._exp_provider.load(self._mol_id)
        if self._exp_provider.can_bootstrap:
            exp_data, exp_bootstrap_data = self._exp_provider.bootstrap(
                self._mol_id, self.num_bootstrap_curves
            )
            # Virtual titration curves
            exp_curves = np.asarray([curve.mean_charge for curve in exp_bootstrap_data])
        return exp_data, exp_curves, exp_bootstrap_data

    def save_all(self, dir: str, ext="pdf"):
        """Save all figures.

        Parameters
        ----------
        dir - output directory for all files
        ext - Extension of the images.
        """
        if not os.path.isdir(dir):
            os.makedirs(dir)
        for desc, method in self._figures.items():
            for figtype, figure in method.items():
                figure.savefig(
                    os.path.join(dir, f"{desc}-{figtype}-{self._mol_id}.{ext}")
                )

        copyfile(self._mol_img, os.path.join(dir, f"{self._mol_id}-molecule.{ext}"))

        with open(os.path.join(dir, f"report-{self._mol_id}.tex"), "w") as latexfile:
            latexfile.write(self._tex_source)

    def generate_latex(self, img_ext="pdf") -> None:
        """Make a minipage latex document layout containing figures"""

        blocks: List[TexBlock] = list()
        header = ReportHeader(
            self._mol_id,
            [
                meth.method_desc
                for meth in [self._exp_provider] + self._prediction_providers
            ],
            img_ext=img_ext,
        )
        blocks.append(header)
        blocks.append(OverviewRow())
        for id in header.ids:
            blocks.append(MethodResultRow(id))
        blocks.append(ReportFooter())

        for block in blocks:
            self._tex_source += block.render_source()

    def close(self) -> None:
        """Close all figures contained within this reporter to save memory."""
        for desc, method in self._figures.items():
            for figtype, figure in method.items():
                plt.close(figure)
        return

    @classmethod
    def _newfig(cls) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
        # Ensure style before starting figure
        sns.set_style("ticks")
        font = {"size": 11}
        matplotlib.rc("font", **font)
        return plt.subplots(
            1, 1, figsize=cls._figprops["figsize"], dpi=cls._figprops["dpi"]
        )

    @classmethod
    def _add_virtual_titration_bootstrap_sd(
        cls,
        fig_ax: Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes],
        label: str,
        curve: TitrationCurveType,
        bootstrap_curves: np.ndarray,
        color_idx: int,
        perc: float = 5,
        fill=False,
        linestyle="-",
        alpha=1.0,
    ) -> None:
        """Plot the estimate and 2 standard deviations from a bootstrap set in existing fig and axes.

        Parameters
        ----------
        fig_ax - figure and corresponding axes to add lines to
        label - label for plot, used for legend
        curve - TitrationCurve object containing the mean, and the pH values
        bootstrap_curves - 2D array of floats, bootstrap titration curves, with the 0 axis being the different curves, and the 1 axis the pH values.
        ph_values - 1d array the ph values that each point corresponds to.
        color_idx - integer index for picking color from class array `extra_colors`
        perc - percentile, and 100-percentile to plot
            default 5, so 5th and 95th are plotted.
        fill - fill the area between percentiles with color.
        """
        color = cls._figprops["extra_colors"][color_idx]
        std = np.std(bootstrap_curves, axis=0)
        ph_values = curve.ph_values
        mean = curve.mean_charge
        # Unpack tuple
        fig, ax = fig_ax
        ax.plot(
            ph_values,
            mean,
            linestyle,
            linewidth=0.75,
            color=color,
            alpha=alpha,
            label=label,
        )
        ax.plot(
            ph_values,
            mean + (2 * std),
            ":",
            linewidth=0.75,
            color=color,
            alpha=0.5 * alpha,
        )
        ax.plot(
            ph_values,
            mean - (2 * std),
            ":",
            linewidth=0.75,
            color=color,
            alpha=0.5 * alpha,
        )
        if fill:
            ax.fill_between(
                ph_values,
                mean - (2 * std),
                mean + (2 * std),
                facecolor=color,
                alpha=0.1,
            )

        return

    def plot_virtual_titration(
        self,
        exp_loader: SAMPL6DataProvider,
        pred_loader: Optional[SAMPL6DataProvider] = None,
        fig_ax: Optional[Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]] = None,
        index: int = None,
    ):
        """Plot titration curve using the mean charge."""
        if fig_ax is None:
            fig, ax = self._newfig()
        else:
            fig, ax = fig_ax
        # Experiment a black dotted curve, prediction is black solid
        exp_data = deepcopy(exp_loader.load(self._mol_id))
        if pred_loader is None:
            ls = 0
        else:
            pred_data = deepcopy(pred_loader.load(self._mol_id))
            ls = 1
            exp_data.align_mean_charge(pred_data, area_between_curves, self._dpH)

            area = area_between_curves(
                pred_data.mean_charge, exp_data.mean_charge, self._dpH
            )

        ax.plot(
            exp_data.ph_values,
            exp_data.mean_charge,
            color="#333333",
            ls=self._figprops["line_styles"][3],
        )
        if pred_loader is not None:
            ax.plot(
                pred_data.ph_values,
                pred_data.mean_charge,
                color="#333333",
                ls=self._figprops["line_styles"][0],
            )
            # Area between curves is colored in gray
            ax.fill_between(
                pred_data.ph_values,
                exp_data.mean_charge,
                pred_data.mean_charge,
                facecolor=self._figprops["extra_colors"][index],
                interpolate=True,
                alpha=0.7,
            )

            ax.set_title(r"$\Delta$ area : {:.2f}".format(area))

        # Integer labels for y axis
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        # ensure at least one integer unit of charge on axis + .1 for spacing
        ymin, ymax = ax.get_ylim()

        round_min = round(ymin) - 0.05
        round_max = round(ymax) + 0.05
        if ymax < round_max:
            ymax = round_max
        if ymin > round_min:
            ymin = round_min
        ax.set_ylim([ymin, ymax])

        # WITH labels on y axis, but indicate the integer values with ticks
        labels = [item.get_text() for item in ax.get_yticklabels()]
        # empty_string_labels = [""] * len(labels)
        # ax.set_yticklabels(empty_string_labels)
        ax.set_ylabel(r"$Q_\mathsf{avg}$")
        ax.set_xlabel("pH")
        # x-tick every 2 pH units
        ax.set_xticks(np.arange(2.0, 14.0, 2.0))
        # remove top and right spines
        sns.despine()
        # fit everything within bounds
        fig.tight_layout()
        return fig

    def plot_predicted_free_energy(
        self, pred_loader: SAMPL6DataProvider
    ) -> matplotlib.figure.Figure:
        """Plot titration curve using free energies."""
        # colored by number of protons bound
        fig, ax = self._newfig()
        pred_data = pred_loader.load(self._mol_id)
        for i, state_id in enumerate(pred_data.state_ids):
            charge = pred_data.charges[i]
            color = self._figprops["colors_per_charge"][charge]
            # neutral on top
            zorder = 10 - abs(charge)

            ls = 0

            ax.plot(
                pred_data.ph_values,
                pred_data.free_energies[i],
                ls=self._figprops["line_styles"][ls],
                color=color,
                label="n={}".format(charge),
                zorder=zorder,
            )

        ax.set_ylabel(r"Free energy ($k_B T$)")
        ax.set_xlabel("pH")
        ax.set_xticks(np.arange(2.0, 14.0, 2.0))
        # remove top and right spines
        sns.despine(ax=ax)
        # fit everything within bounds
        fig.tight_layout()

        return fig

    def plot_predicted_population(
        self, pred_loader: TitrationCurveType
    ) -> matplotlib.figure.Figure:
        """Plot titration TitrationCurve using free energies."""
        # colored by number of protons bound
        pred_data = pred_loader.load(self._mol_id)
        fig, ax = self._newfig()
        for i, state_id in enumerate(pred_data.state_ids):
            charge = pred_data.charges[i]
            color = self._figprops["colors_per_charge"][charge]
            linestyle = 0

            # Neutral on top
            zorder = 10 - abs(charge)
            ax.plot(
                pred_data.ph_values,
                pred_data.populations[i],
                ls=self._figprops["line_styles"][linestyle],
                color=color,
                label="n={}".format(charge),
                zorder=zorder,
            )

        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_ylim([-0.05, 1.05])
        labels = [item.get_text() for item in ax.get_yticklabels()]
        empty_string_labels = [""] * len(labels)
        ax.set_yticklabels(empty_string_labels)
        ax.set_ylabel("Population")
        ax.set_xlabel("pH")
        ax.set_xticks(np.arange(2.0, 14.0, 2.0))

        # remove top and right spines
        sns.despine(ax=ax)
        # fit everything within bounds
        fig.tight_layout()

        return fig

    def plot_experimental_free_energy(
        self, exp_loader: SAMPL6DataProvider
    ) -> matplotlib.figure.Figure:
        # colored by number of protons bound
        fig, ax = self._newfig()
        exp_data = exp_loader.load(self._mol_id)
        for i, state_id in enumerate(exp_data.state_ids):
            nbound = exp_data.charges[i]
            color = self._figprops["colors_per_charge"][nbound]
            if nbound == 0:
                zorder = 10
            else:
                zorder = 2
            ax.plot(
                exp_data.ph_values,
                exp_data.free_energies[i],
                ls=self._figprops["line_styles"][0],
                color=color,
                label="n={}".format(nbound),
            )

        ax.set_ylabel(r"Free energy ($k_B T$)")
        ax.set_xlabel("pH")
        ax.set_xticks(np.arange(2.0, 14.0, 2.0))
        # remove top and right spines
        sns.despine(ax=ax)
        # fit everything within bounds
        fig.tight_layout()

        return fig


def get_percentiles(array, percentiles):
    nums = list()
    for q in percentiles:
        nums.append(np.percentile(array, q, axis=0))
    return nums


def plot_quantiles(
    curves: np.ndarray, ph_range: np.ndarray, color: str, perc: float = 5, fill=True
):
    """Plot the median, and outer percentiles.

    Parameters
    ----------
    curves - 2D array of bootstrap titration curves, with the 0 axis being the different curves, anx the 1 axis the pH values.
    ph_range - the ph values that each point corresponds to.
    color - a matplotlib color for the elements in the plot
    perc - percentile, and 100-percentile to plot
        default 5, so 5th and 95th are plotted.
    fill - fill the area between percentiles with color.
    """
    quantiles = get_percentiles(curves, [50.0, perc, 100.0 - perc])
    plt.plot(ph_range, quantiles[0], "-", color=color, alpha=1.0, label="median")
    plt.plot(
        ph_range,
        quantiles[1],
        ":",
        color=color,
        alpha=1.0,
        label="{:.0f}th/{:.0f}th percentile".format(perc, 100 - perc),
    )
    plt.plot(ph_range, quantiles[2], ":", color=color, alpha=1.0)
    if fill:
        plt.fill_between(
            ph_range, quantiles[2], quantiles[1], facecolor=color, alpha=0.1
        )


def plot_mean_twosigma(curves: np.ndarray, ph_range: np.ndarray, color: str, fill=True):
    """Plot the mean, plus/minus 2 sigma.

    Parameters
    ----------
    curves - 2D array of bootstrap titration curves, with the 0 axis being the different curves, anx the 1 axis the pH values.
    ph_range - the ph values that each point corresponds to.
    color - a matplotlib color for the elements in the plot
    fill - fill the area between +/- 2 sigma with color.
    """
    mean = np.mean(curves, axis=0)
    std = np.std(curves, axis=0)
    plt.plot(ph_range, mean, "-", color=color, label="mean")
    plt.plot(
        ph_range, mean + 2 * std, ":", alpha=1.0, color=color, label=r"$\pm$2$\sigma$"
    )
    plt.plot(ph_range, mean - 2 * std, ":", alpha=1.0, color=color)
    if fill:
        plt.fill_between(
            ph_range, mean + 2 * std, mean - 2 * std, facecolor=color, alpha=0.1
        )


def plot_subset(curves, ph_range, n_choices: int, color="gray", alpha=0.1):
    """Plot a subset of bootstrap samples.

    Parameters
    ----------
    curves - 2D array of bootstrap titration curves, with the 0 axis being the different curves, anx the 1 axis the pH values.
    ph_range - the ph values that each point corresponds to.
    n_choices - number of samples to plot
    color - a matplotlib color for the elements in the plot
    alpha - transparency of the curves.

    """
    choices = np.random.choice(curves.shape[0], n_choices, replace=False)
    for i in choices:
        plt.plot(ph_range, curves[i], "-", color=color, zorder=0, alpha=alpha)


def plot_correlation_analysis(
    dataframe: pd.DataFrame,
    xlabel: str,
    ylabel: str,
    title: str,
    color: str,
    marker: str,
    error_color="black",
    facecolor="none",
    shaded=True,
    insets=True,
):
    """Plot correlation between experiment and prediction.

    Parameters
    ----------
    dataframe - a typeI/typeIII pKa dataframe
        Has columns "Experimental" , "Experimental SEM" ,"Predicted", and "Predicted SEM"
    title - to put above plot. use '' (empty string) for no title.
    color - edge color of the markers. This plot uses open markers.
    error_color - color of the error bars
    facecolor - color of the face of markers
    """

    # plt.clf()
    fig = plt.figure(figsize=[2.5, 2.5], dpi=150)
    ax = plt.gca()
    ax.set_title(title, fontsize=9)

    # If possible at least show 0, 14 but allow for larger axes
    limit_axes = True

    if (
        np.any(0 > dataframe["pKa Method1"])
        or np.any(16.0 < dataframe["pKa Method1"])
        or np.any(0 > dataframe["pKa Method2"])
        or np.any(16.0 < dataframe["pKa Method2"])
    ):
        limit_axes = False

    ax.errorbar(
        dataframe["pKa Method1"],
        dataframe["pKa Method2"],
        xerr=dataframe["pKa SEM Method1"],
        yerr=dataframe["pKa SEM Method2"],
        fmt="none",
        color=error_color,
        alpha=0.8,
        linewidth=0.5,
        zorder=1,
    )
    ax.scatter(
        dataframe["pKa Method1"],
        dataframe["pKa Method2"],
        marker=marker,
        color=color,
        facecolors=facecolor,
        edgecolors=color,
        alpha=0.8,
        linewidth=0.7,
        zorder=0,
    )

    texts = []

    for r, row in dataframe.iterrows():
        if abs(row.Delta) > 2:
            texts.append(
                ax.text(
                    row["pKa Method1"],
                    row["pKa Method2"],
                    row.Molecule,
                    va="center",
                    ha="center",
                    fontsize=8,
                    zorder=2,
                )
            )

    adjust_text(texts, arrowprops=dict(arrowstyle="->", color="black", zorder=2))

    ax.set_ylabel(ylabel, fontsize=8)
    ax.set_xlabel(xlabel, fontsize=8)

    # enforce limits before linear parts a plotted
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    lims = [min([xlim[0], ylim[0]]), max([xlim[1], ylim[1]])]

    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.xaxis.set_major_locator(MultipleLocator(2.0))
    ax.yaxis.set_major_locator(MultipleLocator(2.0))

    if limit_axes:
        ax.set_xlim([0, 16])
        ax.set_ylim([0, 16])

    plt.tight_layout()
    sns.despine(fig)

    # Add linear guides for 1 and 2 pK unit deviation
    ax.plot((-50.0, 50.0), (-50.0, 50.0), "k", zorder=-1, linewidth=0.5, alpha=0.5)
    ax.plot(
        (-52.0, 48.0),
        (-50.0, 50.0),
        "gray",
        linestyle="--",
        zorder=-1,
        linewidth=0.5,
        alpha=0.5,
    )
    ax.plot(
        (-48.0, 52.0),
        (-50.0, 50.0),
        "gray",
        linestyle="--",
        zorder=-1,
        linewidth=0.5,
        alpha=0.5,
    )
    if shaded:
        ax.fill_between(
            [-50.0, 50.0], [-51.0, 49.0], [-49.0, 51.0], color="gray", alpha=0.1
        )

    return fig, ax


class FullpKaComparison:
    """Compile a full report of pKa mapping analysis across all of the SAMPL6 pKa molecules."""

    _loss_functions = {"square": squared_loss}

    _correlation_metrics = {
        "RMSE": array_rmse,
        "Mean abs. error": array_mae,
        r"pearson $\rho$": wrap_pearsonr,
        "Median abs. error": array_median_error,
    }

    # algorithms per data type
    _mapping_algorithms = dict(
        typeiii={
            "closest": closest_pka,
            "hungarian": hungarian_pka,
            "align": align_pka,
        },
        typei={"closest": closest_pka, "hungarian": hungarian_pka},
        exp={"closest": closest_pka, "hungarian": hungarian_pka, "align": align_pka},
        typeimacro={
            "closest": closest_pka,
            "hungarian": hungarian_pka,
            "align": align_pka,
        },
    )

    def __init__(
        self,
        exp_provider: SAMPL6DataProvider,
        data_providers: List[SAMPL6DataProvider],
        included_molecules: Optional[List[str]] = None,
        n_bootstrap_correlation=5000,
    ):

        """Compile a full report of pKa mapping analysis across all of the SAMPL6 pKa molecules."""

        # TODO this is commented out for debugging, please put check back in in final version.
        # if "exp" != exp_provider.data_type:
        #    raise TypeError("Need an experimental provider as data type")

        self._exp_provider = exp_provider
        self._providers = data_providers
        # Take all the sampl6 molecules by default if no names provided
        self.included_molecules = (
            ["SM{:02d}".format(molecule + 1) for molecule in range(24)]
            if included_molecules is None
            else included_molecules
        )

        self._pka_data = pd.DataFrame()
        self._correlation_df = pd.DataFrame()

        for provider in self._providers:
            if provider.data_type == "exp":
                warnings.warn(
                    "An experiment was provided as a prediction.", UserWarning
                )

        # number of samples for correlation bootstrap analysis
        self._n_bootstrap_correlation = n_bootstrap_correlation

    def analyze_all(self):
        """Calculate all possible pKa mappings es for all molecules and methods"""
        all_providers: List[SAMPL6DataProvider] = [self._exp_provider] + self._providers

        pbar1 = tqdm(all_providers, desc="Dataset", unit="data set")
        for provider1 in pbar1:
            pbar2 = tqdm(all_providers, desc="Dataset2", unit="data set", leave=False)
            for provider2 in pbar2:
                if provider1 == provider2:
                    continue
                pkamap = self._perform_pka_maps(provider1, provider2)
                self._pka_data = self._pka_data.append(
                    pkamap, ignore_index=True, sort=False
                )

        self._correlation_df = self._calculate_correlations()

    @staticmethod
    def _extract_pka_df(titrationcurve: HaspKaType) -> pd.DataFrame:
        """Extract pKa values and standard errors from a TitrationCurve class that has pKa values."""
        return pd.DataFrame({"pKa": titrationcurve.pkas, "SEM": titrationcurve.sems})

    def _perform_pka_maps(
        self, provider1: SAMPL6DataProvider, provider2: SAMPL6DataProvider
    ):
        full_df = pd.DataFrame()
        for mol in tqdm(
            self.included_molecules, desc="pKa maps", unit="molecules", leave=False
        ):
            exp = provider1.load(mol)
            comp = provider2.load(mol)
            exp_pka = self._extract_pka_df(exp)
            comp_pka = self._extract_pka_df(comp)

            # name, function
            for alg, f in self._mapping_algorithms[provider2.data_type].items():
                # name, function
                if alg not in self._mapping_algorithms[provider1.data_type]:
                    continue
                for loss, l in self._loss_functions.items():
                    row_df = f(exp_pka, comp_pka, l)
                    row_df["Algorithm"] = alg
                    row_df["Loss function"] = loss
                    row_df["Molecule"] = mol
                    row_df["Type1"] = provider1.data_type
                    row_df["Method1"] = provider1.label
                    row_df["Method2"] = provider2.label
                    row_df["Type2"] = provider2.data_type
                    full_df = full_df.append(row_df, ignore_index=True, sort=False)

        # Patch dataframe column names
        # Default labels first method as experiment and second as prediction
        full_df = full_df.rename(
            columns={
                "Experimental": "pKa Method1",
                "Experimental SEM": "pKa SEM Method1",
                "Predicted": "pKa Method2",
                "Predicted SEM": "pKa SEM Method2",
            }
        )

        full_df["Delta"] = full_df.apply(
            lambda row: (
                ufloat(row["pKa Method2"], row["pKa SEM Method2"])
                - ufloat(row["pKa Method1"], row["pKa SEM Method1"])
            ),
            axis=1,
        )

        return full_df

    def _calculate_correlations(self):
        """Calculate correlation metrics from pKa mapping dataframes using bootstrap analysis."""
        # name

        correlation_df = pd.DataFrame()

        nonan = self._pka_data.dropna()
        for (method1, method2, algorithm, loss), group in tqdm(
            nonan.groupby(["Method1", "Method2", "Algorithm", "Loss function"]),
            desc="Comparison",
            leave=True,
        ):
            samples = []
            for i in trange(
                self._n_bootstrap_correlation,
                desc="Bootstrap",
                unit="sample",
                leave=False,
            ):
                # Draw new dataframe by bootstrapping over rows
                bootstrap_df = bootstrap_pKa_dataframe(group)
                pkas1 = bootstrap_df["pKa Method1"]
                pkas2 = bootstrap_df["pKa Method2"]
                samples.append([pkas1, pkas2])

            for metric, m in self._correlation_metrics.items():
                estimate = m(group["pKa Method1"], group["pKa Method2"])
                bootstrap_estimates = [m(meth1, meth2) for meth1, meth2 in samples]
                correlation_df = correlation_df.append(
                    {
                        "Algorithm": algorithm,
                        "Metric": metric,
                        "Method1": method1,
                        "Method2": method2,
                        "Loss function": loss,
                        "Value": BootstrapDistribution(
                            estimate, np.asarray(bootstrap_estimates)
                        ),
                    },
                    ignore_index=True,
                    sort=False,
                )

        return correlation_df

    def plot_correlation(self):
        """Make plot of computed versus measured pKa values."""
        figures = dict()
        pal = sns.color_palette("dark")
        labels = [prov.label for prov in self._providers]
        nonan = self._pka_data.dropna()
        for (method1, method2, algorithm, loss), group in tqdm(
            nonan.groupby(["Method1", "Method2", "Algorithm", "Loss function"]),
            desc="Comparison",
            leave=True,
        ):

            if method1 == "Experiment" and method2 != "Experiment":
                facecolor = pal[labels.index(method2)]
            elif method2 == "Experiment" and method1 != "Experiment":
                facecolor = pal[labels.index(method1)]
            else:
                facecolor = "black"

            fig, ax = plot_correlation_analysis(
                group,
                method1 + " pKa",
                method2 + " pKa",
                "",
                "gray",
                "s",
                "gray",
                facecolor=facecolor,
            )
            figures[(method1, method2, algorithm, loss)] = fig

        return figures

    def plot_distribution(self, algorithm):
        """Make plot of deltas between computed versus measured pKa values."""

        jointfig, jointax = plt.subplots(1, 1, figsize=(2.5, 2.5), dpi=150)
        figures = dict()
        pal = sns.color_palette("dark")
        labels = [prov.label for prov in self._providers]
        nonan = self._pka_data.dropna()
        for (method1, method2, algo, loss), group in tqdm(
            nonan.groupby(["Method1", "Method2", "Algorithm", "Loss function"]),
            desc="Comparison",
            leave=True,
        ):

            if algo != algorithm:
                continue
            if method1 == "Experiment" and method2 != "Experiment":
                color = pal[labels.index(method2)]
            elif method2 == "Experiment" and method1 != "Experiment":
                color = pal[labels.index(method1)]
            else:
                color = "black"

            fig, ax = plt.subplots(1, 1, figsize=(2.5, 2.5), dpi=150)
            delta = group.Delta.apply(lambda val: val.nominal_value)
            sns.distplot(
                delta, hist=False, rug=True, kde_kws={"shade": True}, color=color, ax=ax
            )

            ylims = ax.get_ylim()

            xlims = ax.get_xlim()
            if xlims[0] > -5.5 and xlims[1] < 5.5:
                ax.set_xlim([-5, 5])

            texts = []
            for r, row in group.iterrows():
                if abs(row.Delta) > 2:
                    texts.append(
                        ax.text(
                            row.Delta.nominal_value,
                            0.1 * ylims[1],
                            row.Molecule,
                            va="center",
                            ha="center",
                            fontsize=8,
                            zorder=2,
                        )
                    )

            adjust_text(texts)

            if method1 == "Experiment":
                sns.distplot(
                    delta,
                    hist=False,
                    rug=False,
                    color=color,
                    ax=jointax,
                    label=f"{method2}",
                )

            plt.axvline(np.median(delta), lw=1, color="black")
            # Every 0.5 add a tick
            ax.xaxis.set_major_locator(MultipleLocator(1.0))

            plt.xlabel(rf"$pKa$ {method2} - $pKa$ {method1}")
            plt.yticks([])
            plt.tight_layout()
            sns.despine(left=True)
            figures[(algorithm, (method1, method2))] = fig

        jointax.set_yticks([])
        jointax.xaxis.set_major_locator(MultipleLocator(1.0))
        jointax.legend(fontsize=8)
        jointax.set_xlabel(r"$pKa$ computed - $pKa$ experiment")
        jointxlims = jointax.get_xlim()
        if jointxlims[0] > -5.5 and jointxlims[1] < 5.5:
            jointax.set_xlim([-5, 5])
        sns.despine(ax=jointax, left=True)
        jointfig.tight_layout()
        figures[(algorithm, ("overview", ""))] = jointfig
        return figures

    def table_pka(self, algorithm):
        """Produce pKa table for a particular algorithm."""
        df = self._pka_data.dropna()
        df = df[(df.Method1 == "Experiment") & (df.Algorithm == algorithm)]
        labels = [prov.label for prov in self._providers]
        observed_labels = []
        table = None
        for label in labels:
            subset = df[df.Method2 == label]
            if subset.size == 0:
                continue
            observed_labels.append(label)

            if table is None:
                table = subset[["Molecule", "pKa Method1", "pKa SEM Method1"]]
            newset = subset[
                [
                    "Molecule",
                    "pKa Method1",
                    "pKa SEM Method1",
                    "pKa Method2",
                    "pKa SEM Method2",
                    "Delta",
                ]
            ]

            table = table.merge(
                newset, on=["Molecule", "pKa Method1", "pKa SEM Method1"]
            )
            table[label] = table.apply(
                lambda row: ufloat(row["pKa Method2"], row["pKa SEM Method2"]), axis=1
            )
            table = table.rename(columns={"Delta": label + " Delta"})
            table = table.drop(columns=["pKa Method2", "pKa SEM Method2"])

        table["Experiment"] = table.apply(
            lambda row: ufloat(row["pKa Method1"], row["pKa SEM Method1"]), axis=1
        )

        delta_in_order = [l + " Delta" for l in observed_labels]

        cols = [
            label for pair in zip(observed_labels, delta_in_order) for label in pair
        ]
        cols.insert(0, "Experiment")
        for col in cols:
            table[col] = table[col].apply(to_str)
        table = table[["Molecule", *cols]]

        # table = table.rename(columns={"Molecule": "{Molecule}"})

        # Column names need to be wrapped for siunitx table
        table.columns = [f"{{{col}}}" for col in table.columns]

        tex_syntax = table.to_latex(escape=False, index=False)
        # regex grabs the line with specification of column types
        colspec_finder = r"\\begin{tabular}{.*}\n"
        # Exp has two decimals on the uncertainty and one sig digit
        exp_col_type = "S[table-format=-1.2,table-figures-uncertainty=1]"
        # Prediction typically has only one decimal on the uncertainty
        pred_col_type = "S[table-format=-1.1,table-figures-uncertainty=1]"
        tex_syntax = re.sub(
            colspec_finder,
            f"\\\\begin{{tabular}}{{c{exp_col_type}{(len(df.columns)-1) * pred_col_type}}}",
            tex_syntax,
            0,
        )
        tex_table = (
            "\\sisetup{separate-uncertainty=true}\n"
            "\\begin{table}\n"
            "\\centering\n"
            f"{tex_syntax}"
            "\\end{table}\n"
        )

        return tex_table

    def table_correlations(self, algorithm):
        """Return overall correlation/error statistics for pKa comparison."""
        selection = (self._correlation_df["Method1"] == "Experiment") & (
            self._correlation_df["Algorithm"] == algorithm
        )
        subset = self._correlation_df[selection]
        subset = subset.rename(columns={"Method2": "Method"})
        return subset[["Method", "Metric", "Value"]].to_latex(index=True, escape=False)


class TitrationComparison:
    """Compile a full report of titration curve analysis across all of the SAMPL6 pKa molecules."""

    _curve_metrics = {"area": area_curve_vectorized, "rmsd": rmsd_curve_vectorized}

    # default interval between pH
    _dpH = 0.1

    def __init__(
        self,
        exp_provider: SAMPL6DataProvider,
        data_providers: List[SAMPL6DataProvider],
        included_molecules: Optional[List[str]] = None,
        n_bootstrap_titration=100,
    ):
        """Compile a full report of pKa mapping analysis across all of the SAMPL6 pKa molecules."""

        if "exp" != exp_provider.data_type:
            raise TypeError("Need an experimental provider as data type")

        self._exp_provider = exp_provider
        self._providers = data_providers
        # Take all the sampl6 molecules by default if no names provided
        self.included_molecules = (
            ["SM{:02d}".format(molecule + 1) for molecule in range(24)]
            if included_molecules is None
            else included_molecules
        )

        self._curve_df = pd.DataFrame()
        self._exp_data = dict()
        self._raw_curves = dict()

        for provider in self._providers:
            if provider.data_type == "exp":
                warnings.warn(
                    "An experiment was provided as a prediction.", UserWarning
                )

        self._n_bootstrap_titration = n_bootstrap_titration

    def analyze_all(self):
        """Calculate all possible pKa mappings, and all area between curves for all molecules and methods"""

        pbar = tqdm(self._providers, desc="Dataset", unit="data set")
        for provider in pbar:
            pbar.set_description(desc=provider.method_desc, refresh=True)
            self._curve_df = self._curve_df.append(
                self._compare_titration_curves(self._exp_provider, provider),
                ignore_index=True,
                sort=False,
            )
        pbar.set_description(desc="Done.", refresh=True)

    def _compare_titration_curves(
        self, exp_provider: SAMPL6DataProvider, computed_provider: SAMPL6DataProvider
    ):
        """Calculate deviation between all titration curves."""
        full_df = pd.DataFrame()

        for mol in tqdm(
            self.included_molecules, desc="Titration", unit="molecule", leave=False
        ):
            log.debug(mol)
            if computed_provider.can_bootstrap:
                if mol not in self._exp_data:
                    exp_data, exp_bootstrap_data = exp_provider.bootstrap(
                        mol, self._n_bootstrap_titration
                    )
                    self._exp_data[mol] = (exp_data, exp_bootstrap_data)
                else:
                    exp_data, exp_bootstrap_data = self._exp_data[mol]

                # move the movable curve to be as close to the target as possible, as estimated by area
                # Virtual titration curves
                pred_data, bootstrap_data = computed_provider.bootstrap(
                    mol, self._n_bootstrap_titration
                )
                exp_curves = np.asarray(
                    [curve.mean_charge for curve in exp_bootstrap_data]
                )
                pred_curves = np.asarray(
                    [curve.mean_charge for curve in bootstrap_data]
                )

                exp_curves = np.vstack([exp_curves, exp_data.mean_charge])
                pred_curves = np.vstack([pred_curves, pred_data.mean_charge])

                for metric, m in tqdm(
                    self._curve_metrics.items(),
                    desc="Delta",
                    unit="metric",
                    leave=False,
                ):
                    log.debug(metric)
                    q_curves_pred, q_curves_exp_fit, scores = fit_titration_curves_3d(
                        pred_curves, exp_curves, m, self._dpH
                    )
                    self._raw_curves[(computed_provider.method_desc, mol, metric)] = (
                        q_curves_pred,
                        q_curves_exp_fit,
                        scores,
                        pred_data.ph_values,
                    )
                    dist = BootstrapDistribution(
                        np.asscalar(scores[-1]), scores[:-1].squeeze()
                    )
                    full_df = full_df.append(
                        dict(
                            Molecule=mol,
                            Metric=metric,
                            Value=dist,
                            Method=computed_provider.method_desc,
                            Type=computed_provider.data_type,
                        ),
                        ignore_index=True,
                        sort=False,
                    )

            else:
                for metric, m in tqdm(
                    self._curve_metrics.items(),
                    desc="Delta",
                    unit="metric",
                    leave=False,
                ):
                    log.debug(metric)
                    exp_data = exp_provider.load(mol)
                    pred_data = computed_provider.load(mol)
                    exp_data.align_mean_charge(pred_data, m, self._dpH)
                    dev = np.asscalar(
                        m(
                            pred_data.mean_charge[:, np.newaxis].T,
                            exp_data.mean_charge[:, np.newaxis].T,
                            self._dpH,
                        )
                    )
                    # single decimal point reported only
                    dev = round(dev * 10) / 10
                    full_df = full_df.append(
                        dict(
                            Molecule=mol,
                            Metric=metric,
                            Value=dev,
                            Method=computed_provider.method_desc,
                            Type=computed_provider.data_type,
                        ),
                        ignore_index=True,
                        sort=False,
                    )
                    self._raw_curves[(computed_provider.method_desc, mol, metric)] = (
                        np.asarray([pred_data.mean_charge]),
                        np.asarray([exp_data.mean_charge]),
                        np.asarray([dev]),
                        pred_data.ph_values,
                    )

        return full_df

    def titration_color_legend(self):
        """Produce legend for colors of titration curves."""
        pal = sns.color_palette("dark")
        fig, ax = plt.subplots(1, 1, figsize=(4, 2), dpi=150)

        ax.plot([0, 1], [0, 1], color="black", lw=3, label="Experiment")

        for p, prov in enumerate(self._providers):
            ax.plot([0, 1], [0, 1], color=pal[p], lw=3, label=prov.label)

        ax.plot([0, 1], [0, 1], color="black", ls="--", lw=3, label="95% bootstrap CI")

        # Separate legend figure
        figlegend, axlegend = plt.subplots(1, 1, figsize=[8, 1], dpi=150)
        handles, labels = ax.get_legend_handles_labels()
        # handles = np.concatenate((handles[::2],handles[1::2]),axis=0)
        # labels = np.concatenate((labels[::2],labels[1::2]),axis=0)

        leg = figlegend.legend(handles, labels, loc="center", ncol=4, frameon=False)
        axlegend.get_xaxis().set_visible(False)
        axlegend.get_yaxis().set_visible(False)
        for spine in ["top", "left", "bottom", "right"]:
            axlegend.spines[spine].set_visible(False)
        plt.close(fig)

        return figlegend

    def plot_titration_curves(self, include_micro=False):
        """Plot the titration curves for all the methods that were analyzed."""
        pal = sns.color_palette("dark")
        figures = dict()
        for mol in tqdm(
            self.included_molecules, desc="Titration", unit="molecule", leave=False
        ):
            fullfig, fullax = plt.subplots(1, 1, figsize=(3, 3), dpi=150)

            for p, prov in enumerate(self._providers):
                sepfig, sepax = plt.subplots(1, 1, figsize=(3, 3), dpi=150)

                if not include_micro:
                    if prov.data_type not in ["typeiii", "typeimacro"]:
                        continue
                computed, experimental, scores, ph_values = self._raw_curves[
                    (prov.method_desc, mol, "area")
                ]

                alpha = 0.9
                color = pal[p]
                std = np.std(computed, axis=0)
                mean = computed[-1, :]
                # Unpack                 tuple
                for i, ax in enumerate([fullax, sepax]):
                    ax.plot(
                        ph_values, mean, "-", linewidth=1.5, color=color, alpha=alpha
                    )

                    ax.plot(
                        ph_values,
                        experimental[-1, :],
                        "-",
                        linewidth=1.5,
                        color="black",
                        alpha=0.7,
                    )

                    ax.plot(
                        ph_values,
                        mean + (2 * std),
                        "--",
                        linewidth=0.75,
                        color=color,
                        alpha=0.5 * alpha,
                    )
                    ax.plot(
                        ph_values,
                        mean - (2 * std),
                        "--",
                        linewidth=0.75,
                        color=color,
                        alpha=0.5 * alpha,
                    )
                    if i == 1:
                        ax.fill_between(
                            ph_values,
                            mean - (2 * std),
                            mean + (2 * std),
                            facecolor=color,
                            alpha=0.05,
                        )

                sepax.yaxis.set_major_locator(MaxNLocator(integer=True))
                # No labels on y axis, but indicate the integer values with ticks
                # labels = [item.get_text() for item in ax.get_yticklabels()]
                # empty_string_labels = [""] * len(labels)
                # ax.set_yticklabels(empty_string_labels)
                sepax.set_ylabel(r"$Q_\mathsf{avg}$")
                sepax.set_xlabel("pH")
                # x-tick every 2 pH units
                sepax.set_xticks(np.arange(2.0, 14.0, 2.0))
                # remove top and right spines
                sns.despine(sepfig)
                title = f"{prov.method_desc} {mol}"
                sepax.set_title(title)
                # fit everything within bounds
                sepfig.tight_layout()
                figures[title] = sepfig

            fullax.yaxis.set_major_locator(MaxNLocator(integer=True))
            # No labels on y axis, but indicate the integer values with ticks
            # labels = [item.get_text() for item in ax.get_yticklabels()]
            # empty_string_labels = [""] * len(labels)
            # ax.set_yticklabels(empty_string_labels)
            fullax.set_ylabel(r"$Q_\mathsf{avg}$")
            fullax.set_xlabel("pH")
            # x-tick every 2 pH units
            fullax.set_xticks(np.arange(2.0, 14.0, 2.0))
            # remove top and right spines
            sns.despine(fullfig)
            fullax.set_title(mol)
            # fit everything within bounds
            fullfig.tight_layout()

            figures[f"overview {mol}"] = fullfig

        return figures

    def table_overall_performance(self):
        """Compile the general overview table."""
        # Individual assessments are stored as distributions, but for aggregation, use floats.

        self._curve_df["Value_float"] = self._curve_df["Value"].apply(float)
        aggregate_df = pd.DataFrame()

        # average over  all molecules
        for (method, metric, stype), group in self._curve_df.groupby(
            ["Method", "Metric", "Type"]
        ):
            aggregate_df = aggregate_df.append(
                dict(
                    Method=method,
                    Metric=metric,
                    Type=stype,
                    Algorithm="Titration curve",
                    Value=bootstrapped_func(group["Value_float"], 10000, np.mean),
                ),
                ignore_index=True,
                sort=False,
            )

        aggregate_df = aggregate_df.append(self._corr_df, ignore_index=True, sort=False)

        # Every method is a row.
        rownames = [prov.method_desc for prov in self._providers]
        # used to find row indices in table
        row_key = list(set(rownames))

        # Every comparison algorithm/method e.g. Hungarian, closest, or titration curve gets a main column
        # Type iii used here because it should have all types, whereas typei doesnt have align
        column_key = list(self._mapping_algorithms["typeiii"].keys())
        column_key.append("Titration curve")

        # subcolumn indices as lists in a dict[main column]
        subcolumn_keys = dict()
        for column in column_key:
            if column in self._mapping_algorithms["typeiii"]:
                subcolumn_keys[column] = list(self._correlation_metrics.keys())
            elif column == "Titration curve":
                subcolumn_keys[column] = list(self._curve_metrics.keys())
            else:
                raise ValueError(f"Unexpected column encountered: {column}.")

        # Number of rows is one row per method and two label rows
        num_rows = len(row_key) + 2
        # The number of actual columns in the table is the number of subcolumns and one label column
        num_columns = sum([len(col) for col in subcolumn_keys.values()]) + 1
        # Array of strings shorter than 32 unicode characters
        table = np.empty((num_rows, num_columns), dtype="<U32")

        # em dash for anything with no value
        table[2:, 1:] = " \\textemdash "
        # Label rows in table
        for row_idx, rowname in enumerate(row_key, start=2):
            table[row_idx, 0] = rowname

        # label columns in table
        offset = 1
        for col_idx, (colname, subcols) in enumerate(subcolumn_keys.items(), start=1):
            table[0, offset] = colname
            for subcol_idx, subcolname in enumerate(subcols, start=offset):
                table[1, subcol_idx] = subcolname
            offset += len(subcols)

        # insert value in cells
        for ix, cell in aggregate_df.iterrows():
            # skip these for now
            if cell["Loss function"] == "abs":
                continue
            val = str(cell.Value)
            row_idx = row_key.index(cell.Method) + 2
            subcol_idx = 1
            for colname, subcols in subcolumn_keys.items():
                if colname == cell.Algorithm:
                    subcol_idx += subcols.index(cell.Metric)
                    break
                elif colname != cell.Algorithm:
                    subcol_idx += len(subcols)
            #     print(cell, row_idx, subcol_idx)
            table[row_idx, subcol_idx] = val

        alignment_string = "{}".format(num_columns * "c")

        content = ""
        for line in table:
            content += " & ".join(line) + "\\\\\n"

        return LatexTable(alignment_string, content).render_source()

    def table_curve_area(self):
        return (
            self._curve_df[self._curve_df.Metric == "area"]
            .pivot(index="Molecule", columns="Method", values="Value")[
                [p.method_desc for p in self._providers]
            ]
            .to_latex()
        )

    def table_curve_rmsd(self):
        return (
            self._curve_df[self._curve_df.Metric == "rmsd"]
            .pivot(index="Molecule", columns="Method", values="Value")[
                [p.method_desc for p in self._providers]
            ]
            .to_latex()
        )


class LatexTable(TexBlock):
    """Basic class for latex syntax block, to be added to report."""

    tex_src = (
        "\\begin{{{sideways}table}}\n"
        "\\centering\n"
        "\\begin{{tabular}}{{{alignment}}}\n"
        "{content}\n"
        "\\end{{tabular}}\n"
        "\\end{{{sideways}table}}\n"
    )

    def __init__(self, alignment, content, sideways=False):
        super(LatexTable, self).__init__(alignment=alignment, content=content)
        if sideways:
            self._variables["sideways"] = "sideways"
            warnings.warn(
                "Remember to add \\usepackage{{rotating}} to your preamble", UserWarning
            )
        else:
            self._variables["sideways"] = ""
        return


def plot_micropka_network(
    titrationcurve: TypeIPrediction
) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """Plot the network of microstates connected by pKa values."""
    if not isinstance(titrationcurve, TypeIPrediction):
        raise TypeError(
            "This function only implements handling of TypeIPrediction objects."
        )

    charges = dict(zip(titrationcurve.state_ids, titrationcurve.charges))

    node_colors = []
    for node in titrationcurve.augmented_graph.nodes:
        node_colors.append(charge_colors[charges[node]])

    edge_labels = dict()
    for edge in titrationcurve.augmented_graph.edges:
        # multiply by -1 to make arrow direction match pKa direction
        edge_labels[(edge[0], edge[1])] = "{:.2f}".format(
            -1 * titrationcurve.augmented_graph.edges[edge[0], edge[1]]["pKa"]
        )

    fig = plt.figure(figsize=(9,13), dpi=75)
    pos = pydot_layout(titrationcurve.graph, prog="dot")
    nx.draw_networkx_nodes(
        titrationcurve.augmented_graph,
        pos,
        node_color=node_colors,
        alpha=0.85,
        node_size=6000,
        node_shape="o",
    )
    nx.draw_networkx_labels(
        titrationcurve.augmented_graph, pos, node_color=node_colors, alpha=1
    )
    nx.draw_networkx_edge_labels(
        titrationcurve.augmented_graph,
        pos,
        label_pos=0.65,
        edge_labels=edge_labels,
        alpha=0.75,
    )
    nx.draw_networkx_edges(
        titrationcurve.augmented_graph,
        pos,
        edge_labels=edge_labels,
        alpha=0.75,
        node_size=6000
    )

    plt.tight_layout()

    sns.palplot(charge_colors.values())
    ax = plt.gca()
    xticks = ax.get_xticks()
    xticks = [x + 0.5 for x in xticks]
    ax.set_xticks(xticks)
    ax.set_xticklabels(["{:+d}".format(l) for l in charge_colors.keys()])
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(16)

    plt.tight_layout()
    return (fig, ax)


def tabulate_cycles(titrationcurve: TypeIPrediction, length: int = 4) -> str:
    """Returns a string with table of all cycles of specified length (default: 4).

        Warning
        -------
        This example implementation uses networkx.simple_cycles. 
        It is not very optimized, and involves making the list of all cycles and 
        then picking out the ones of the correct length. 
        This will be very slow for larger networks.

    """
    # # Find cycles of a specific length
    if not isinstance(titrationcurve, TypeIPrediction):
        raise TypeError(
            "This function only implements handling of TypeIPrediction objects."
        )

    markdown: str = ""
    cycles: List[Tuple[List[str], float]] = []

    for cycle in tqdm(nx.simple_cycles(titrationcurve.augmented_graph), desc="cycles"):
        if len(cycle) == length:
            rotated = deque(cycle)
            rotated.rotate(1)
            tot = 0.0
            for edge in zip(cycle, rotated):
                tot += titrationcurve.augmented_graph.edges[edge]["pKa"]
            cycles.append((cycle, tot))

    markdown += "cycle | sum pKa \n -----|----- \n "

    for (cycle, tot) in cycles:
        markdown += f" {cycle} | {tot:.3f} \n"

    return markdown
