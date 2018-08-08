import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from typing import List, Tuple, Dict, Callable, Union, Optional, Any, TypeVar
from .titrato import TitrationCurve
from .sampl import (
    TypeIIPrediction,
    TypeIPrediction,
    TypeIIIPrediction,
    SAMPL6Experiment,
    TitrationCurveType,
    SAMPL6DataProvider
)
from .stats import area_between_curves
import pandas as pd
import seaborn as sns
from shutil import copyfile
import string

# Default styling
sns.set_style("ticks")
font = {"size": 18}
matplotlib.rc("font", **font)

import os

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
        "\\documentclass[11pt,final]{{article}}\n"
        "\\renewcommand{{\\familydefault}}{{\\sfdefault}}\n"
        "\\usepackage[utf8]{{inputenc}}\n"
        "\\usepackage[english]{{babel}}\n"
        "\\usepackage[showframe]{{geometry}}\n"
        "\\geometry{{letterpaper}}\n"
        "\\geometry{{margin=1in}}\n"
        "\\usepackage{{graphicx}}\n"
        "\n\\begin{{document}}\n"
    )

    # syntax for adding a new variable
    tex_var = "\\newcommand{{\\{}}}{{{}}}\n"

    def __init__(self, mol_id, method_names):
        """Initialize the header of the file by setting all appropriate variables"""
        variables = dict()
        variables["molid"] = mol_id

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
        "\n\\noindent \n"
        "\\begin{{minipage}}[s]{{0.35\\textwidth}}\\centering\n"
        "\\includegraphics[width=\\textwidth]{{\\molid-molecule.png}}\n"
        "\\end{{minipage}}\n"
        "\\begin{{minipage}}[s]{{0.35\\textwidth}}\n"
        "\\includegraphics[width=\\textwidth]{{overview-virtual-titration-\\molid.png}}\n"
        "\\end{{minipage}}\n"
        "\\begin{{minipage}}[s]{{0.23\\textwidth}}\n"
        "\\includegraphics[width=\\textwidth]{{overview-legend-\\molid.png}}\n"
        "\\end{{minipage}}\n"
    )


class MethodResultRow(TexBlock):
    """A row of figures for a single method"""

    tex_src = (
        "\n\\begin{{minipage}}[s]{{\\textwidth}}\\centering\n"
        "{{\\textbf \\method{id}}}\n"
        "\\end{{minipage}}\n"
        "\n\\noindent\n"
        "\\begin{{minipage}}[s]{{0.32\\textwidth}}\\centering\n"
        "\\includegraphics[width=\\textwidth]{{\\method{id}-virtual-titration-\\molid.png}}\n"
        "\\end{{minipage}}\n"
        "\\begin{{minipage}}[s]{{0.32\\textwidth}}\n"
        "\\includegraphics[width=\\textwidth]{{\\method{id}-free-energy-\\molid.png}}\n"
        "\\end{{minipage}}\n"
        "\\begin{{minipage}}[s]{{0.32\\textwidth}}\n"
        "\\includegraphics[width=\\textwidth]{{\\method{id}-populations-\\molid.png}}\n"
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
        "figsize": (3, 3),
        "line_styles": ["-", "--", "-.", ":"],
        # [    0 H,     1 H, 2 H , ... ]
        "color_per_state": [
            "#333333",
            "#00994d",
            "#e60000",
            "#006bb3",
            "#ffcc00",
            "#808080",
            "#8600b3",
            "#ff751a",
            "#00b3b3",
        ],
    }

    def __init__(
        self,
        mol_id: str,
        exp_provider: SAMPL6DataProvider,
        prediction_providers: List[SAMPL6DataProvider],
        mol_png: str,
    ) -> None:
        """Instantiate the analysis from the identifier of the molecule, and providers of the data.
        
        Parameters
        ----------
        mol_id - molecule associated with this report
        exp_provider - provider for the experimental data source
        prediction_provides - list of providers for all the predictions
        mol_png - location where a png image of the molecule can be found
        """

        self._exp_provider = exp_provider
        self._prediction_providers = prediction_providers
        self._figures: Dict[str, Dict[str, matplotlib.figure.Figure]] = {
            pred.method_desc: dict() for pred in prediction_providers
        }
        # Add dict for overview figures
        self._figures["overview"] = dict()
        # Data tables by description, and latex format
        self._tables: Dict[str, str] = dict()
        # Latex Report document
        self._tex_source = ""
        self._num_predictions = len(prediction_providers)
        self._mol_id = mol_id
        self._mol_png = mol_png

        return

    def make_all_plots(self):
        """Make all conceivable plots for every prediction."""

        # All methods tested against the same experimental values.
        exp_data = self._plot_virtual_titration_overview()

        # Each method gets its own plots.
        for pred in self._prediction_providers:

            desc = pred.method_desc
            pred_data = pred.load(self._mol_id)
            ph_values = pred_data.ph_values
            pred_data.align_mean_charge(exp_data, area_between_curves, self._dpH)

            # Virtual titration plot
            figtype = "virtual-titration"
            newfig = SAMPL6ReportGenerator.plot_virtual_titration(exp_data, pred_data)
            self._figures[desc][figtype] = newfig
            # Free enery values
            figtype = "free-energy"
            newfig = SAMPL6ReportGenerator.plot_predicted_free_energy(pred_data)
            self._figures[desc][figtype] = newfig
            # Populations
            figtype = "populations"
            newfig = SAMPL6ReportGenerator.plot_predicted_population(pred_data)
            self._figures[desc][figtype] = newfig            

    def _plot_virtual_titration_overview(self):
        """Plot an overview of all methods using the virtual charge titration.
        
        Also stores a legend with color codes for each method, that can be used with other overview figures.
        """
        # All methods tested against the same experimental values.
        exp_data = self._exp_provider.load(self._mol_id)
        if self._exp_provider.can_bootstrap:
            exp_data, exp_bootstrap_data = self._exp_provider.bootstrap(self._mol_id)
            # Virtual titration curves
            exp_curves = np.asarray([curve.mean_charge for curve in exp_bootstrap_data])

        # Overview charge titration
        titration_fig_ax = self._newfig()

        # Experiment a black solid curve
        self._add_virtual_titration_bootstrap_sd(
            titration_fig_ax, "Experiment", exp_data, exp_curves, 0
        )

        for idx, pred in enumerate(self._prediction_providers, start=1):
            desc = pred.method_desc
            if pred.can_bootstrap:
                pred_data, bootstrap_data = pred.bootstrap(self._mol_id)

                # Align all to experiment curve (note this is a joint bootstrap of experiment and prediction)
                for dat, exp_dat in zip(bootstrap_data, exp_bootstrap_data):
                    dat.align_mean_charge(exp_dat, area_between_curves, self._dpH)
                curves = np.asarray([curve.mean_charge for curve in bootstrap_data])
                self._add_virtual_titration_bootstrap_sd(
                    titration_fig_ax, desc, pred_data, curves, idx
                )
            else:
                pred_data = pred.load(self._mol_id)
                pred_data.align_mean_charge(exp_data, area_between_curves, self._dpH)
                curve = pred_data.mean_charge
                self._add_virtual_titration_bootstrap_sd(
                    titration_fig_ax, desc, pred_data, np.asarray([curve]), idx
                )

        # Unpack tuple.
        fig, ax = titration_fig_ax
        # Integer labels for y axis
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        # No labels on y axis, but indicate the integer values with ticks
        labels = [item.get_text() for item in ax.get_yticklabels()]
        empty_string_labels = [""] * len(labels)
        ax.set_yticklabels(empty_string_labels)
        ax.set_ylabel("Mean charge")
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
        return exp_data

    def save_all(self, ext: str, dir: str):
        """Save all figures.
        
        Parameters
        ----------
        ext - Extension of the file.
        dir - output directory for all files
        """
        if not os.path.isdir(dir):
            os.makedirs(dir)
        for desc, method in self._figures.items():
            for figtype, figure in method.items():
                figure.savefig(
                    os.path.join(dir, f"{desc}-{figtype}-{self._mol_id}.{ext}")
                )

        copyfile(self._mol_png, os.path.join(dir, f"{self._mol_id}-molecule.png"))

        with open(os.path.join(dir, f"report-{self._mol_id}.tex"), "w") as latexfile:
            latexfile.write(self._tex_source)

    def generate_latex(self) -> None:
        """Make a minipage latex document layout containing figures"""

        blocks: List[TexBlock] = list()
        header = ReportHeader(
            self._mol_id, [meth.method_desc for meth in self._prediction_providers]
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
        font = {"size": 18}
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
    ) -> None:
        """Plot the estimate and 2 standard deviations from a bootstrap set in existing fig and axes.
        
        Parameters
        ----------
        fig_ax - figure and corresponding axes to add lines to
        label - label for plot, used for legend
        curve - TitrationCurve object containing the mean, and the pH values
        bootstrap_curves - 2D array of floats, bootstrap titration curves, with the 0 axis being the different curves, and the 1 axis the pH values.
        ph_values - 1d array the ph values that each point corresponds to.
        color_idx - integer index for picking color from class array `color_per_state`   	
        perc - percentile, and 100-percentile to plot 
            default 5, so 5th and 95th are plotted.
        fill - fill the area between percentiles with color.
        """
        color = cls._figprops["color_per_state"][color_idx]
        std = np.std(bootstrap_curves, axis=0)
        ph_values = curve.ph_values
        mean = curve.mean_charge
        # Unpack tuple
        fig, ax = fig_ax
        ax.plot(ph_values, mean, "-", color=color, alpha=1.0, label=label)
        ax.plot(ph_values, mean + (2 * std), ":", color=color, alpha=1.0)
        ax.plot(ph_values, mean - (2 * std), ":", color=color, alpha=1.0)
        if fill:
            ax.fill_between(
                ph_values,
                mean - (2 * std),
                mean + (2 * std),
                facecolor=color,
                alpha=0.1,
            )

        return

    @classmethod
    def plot_virtual_titration(
        cls,
        exp: TitrationCurveType,
        pred: TitrationCurveType,
        fig_ax: Optional[Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]] = None,
    ):
        """Plot titration curve using the mean charge."""
        if fig_ax is None:
            fig, ax = cls._newfig()
        else:
            fig, ax = fig_ax
        # Experiment a black solid curve, prediction is green
        ax.plot(
            pred.ph_values,
            exp.mean_charge,
            color=cls._figprops["color_per_state"][0],
            ls=cls._figprops["line_styles"][1],
        )
        ax.plot(
            pred.ph_values,
            pred.mean_charge,
            color=cls._figprops["color_per_state"][1],
            ls=cls._figprops["line_styles"][0],
        )
        # Area between curves is colored in gray
        ax.fill_between(
            pred.ph_values,
            exp.mean_charge,
            pred.mean_charge,
            facecolor="#808080",
            interpolate=True,
            alpha=0.7,
        )
        # Integer labels for y axis
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        # No labels on y axis, but indicate the integer values with ticks
        labels = [item.get_text() for item in ax.get_yticklabels()]
        empty_string_labels = [""] * len(labels)
        ax.set_yticklabels(empty_string_labels)
        ax.set_ylabel("Mean charge")
        ax.set_xlabel("pH")
        # x-tick every 2 pH units
        ax.set_xticks(np.arange(2.0, 14.0, 2.0))
        # remove top and right spines
        sns.despine()
        # fit everything within bounds
        fig.tight_layout()
        return fig

    @classmethod
    def plot_predicted_free_energy(
        cls, pred: TitrationCurveType, exp: Optional[TitrationCurveType] = None
    ) -> matplotlib.figure.Figure:
        """Plot titration curve using free energies."""
        # colored by number of protons bound
        fig, ax = cls._newfig()
        for i, state_id in enumerate(pred.state_ids):
            nbound = pred.nbound[i]
            color = cls._figprops["color_per_state"][nbound]
            if nbound == 0:
                zorder = 10
            else:
                zorder = 2
            ax.plot(
                pred.ph_values,
                pred.free_energies[i],
                ls=cls._figprops["line_styles"][0],
                color=color,
                label="n={}".format(nbound),
                zorder=zorder,
            )

        if exp is not None:
            for i, state_id in enumerate(exp.state_ids):
                nbound = exp.nbound[i]
                color = cls._figprops["color_per_state"][nbound]
                if nbound == 0:
                    zorder = 10
                else:
                    zorder = 2
                ax.plot(
                    exp.ph_values,
                    exp.free_energies[i],
                    ls=cls._figprops["line_styles"][1],
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

    @classmethod
    def plot_predicted_population(
        cls, pred: TitrationCurveType, exp: Optional[TitrationCurveType] = None
    ) -> matplotlib.figure.Figure:
        """Plot titration TitrationCurve using free energies."""
        # colored by number of protons bound
        fig, ax = cls._newfig()
        for i, state_id in enumerate(pred.state_ids):
            nbound = pred.nbound[i]
            color = cls._figprops["color_per_state"][abs(nbound)]
            # Render negative differently
            if nbound < 0:
                linestyle = 1
            else:
                linestyle = 0

            if nbound == 0:
                zorder = 10
            else:
                zorder = 2
            ax.plot(
                pred.ph_values,
                pred.populations[i],
                ls=cls._figprops["line_styles"][linestyle],
                color=color,
                label="n={}".format(nbound),
                zorder=zorder,
            )

        if exp is not None:
            for i, state_id in enumerate(exp.state_ids):
                nbound = exp.nbound[i]
                color = cls._figprops["color_per_state"][nbound]
                if nbound == 0:
                    zorder = 10
                else:
                    zorder = 2
                ax.plot(
                    exp.ph_values,
                    exp.populations[i],
                    ls=cls._figprops["line_styles"][1],
                    color=color,
                    label="n={}".format(nbound),
                )

        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
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


@classmethod
def plot_experimental_free_energy(
    cls, exp: TitrationCurveType
) -> matplotlib.figure.Figure:
    # colored by number of protons bound
    fig, ax = cls._newfig()

    for i, state_id in enumerate(exp.state_ids):
        nbound = exp.nbound[i]
        color = cls._figprops["color_per_state"][nbound]
        if nbound == 0:
            zorder = 10
        else:
            zorder = 2
        ax.plot(
            exp.ph_values,
            exp.free_energies[i],
            ls=cls._figprops["line_styles"][0],
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
    quantiles = get_percentiles(curves, [50., perc, 100. - perc])
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

