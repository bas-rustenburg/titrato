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
from copy import deepcopy 

# Default styling
sns.set_style("ticks")
glob_font = {"size": 11}
matplotlib.rc("font", **glob_font)

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

    tex_src = ("\\section{{\\molid}}"
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
        "\\includegraphics[scale=1]{{Reports/\\method{id}-virtual-titration-\\molid.\\imgext}}\n"
        "\\end{{minipage}}\n"
        "\\begin{{minipage}}[s]{{0.33\\textwidth}}\n"
        "\\includegraphics[scale=1]{{Reports/\\method{id}-free-energy-\\molid.\\imgext}}\n"
        "\\end{{minipage}}\n"
        "\\begin{{minipage}}[s]{{0.33\\textwidth}}\n"
        "\\includegraphics[scale=1]{{Reports/\\method{id}-populations-\\molid.\\imgext}}\n"
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
        "figsize": (2.0, 2.0), # 3 figures fitting between 3 cm margins on letter paper
        "line_styles": ["-", "--", "-.", ":"],        
        "colors_per_charge": {
            0: "#333333",
            1: "#00b3b3",
            2: "#00994d",            
            3: "#006bb3",            
            4: "#808080",
            -1: "#ffcc00",            
            -2: "#ff751a",            
            -3: "#e60000",
            -4: "#8600b3",            
        },
        # default seaborn color palette
        "extra_colors": sns.color_palette(),
    }

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
        for charge in range(-4,5):
            color = self._figprops["colors_per_charge"][charge]
            ax.plot([0,1],[0,1], color=color, label = f'{charge:d}')
        # Separate legend figure
        figlegend, axlegend = plt.subplots(
            1, 1, figsize=[4,2], dpi=self._figprops["dpi"]
        )
        handles, labels = ax.get_legend_handles_labels()
        # handles = np.concatenate((handles[::2],handles[1::2]),axis=0)
        # labels = np.concatenate((labels[::2],labels[1::2]),axis=0)
             
        leg = figlegend.legend(handles, labels, loc="center",ncol=5)        
        axlegend.get_xaxis().set_visible(False)
        axlegend.get_yaxis().set_visible(False)
        for spine in ["top", "left", "bottom", "right"]:
            axlegend.spines[spine].set_visible(False)
        
        self._figures["overview"]["charge-legend"] = figlegend
        fig.close()

    def make_all_plots(self):
        """Make all available plots for each prediction and the experiment.."""

        self._plot_virtual_titration_overview()
        self._plot_charge_legend()
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
        for pred_loader in self._prediction_providers:

            desc = pred_loader.method_desc            
            # Virtual titration plot
            figtype = "virtual-titration"
            newfig = self.plot_virtual_titration(self._exp_provider, pred_loader=pred_loader)
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
                pred_data.align_mean_charge(exp_data, area_between_curves, self._dpH)
                # Align all to experiment curve (note this is a joint bootstrap of experiment and prediction)
                curves = list()
                for dat, exp_dat in zip(bootstrap_data, exp_bootstrap_data):
                    dat.align_mean_charge(exp_dat, area_between_curves, self._dpH)
                    curves.append(deepcopy(dat.mean_charge))
                curves = np.asarray(curves)
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
            self._mol_id, [meth.method_desc for meth in [self._exp_provider] + self._prediction_providers], img_ext=img_ext
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
        ax.plot(ph_values, mean, "-", linewidth=0.75, color=color, alpha=1.0, label=label)
        ax.plot(ph_values, mean + (2 * std), ":", linewidth=0.75, color=color, alpha=1.0)
        ax.plot(ph_values, mean - (2 * std), ":", linewidth=0.75, color=color, alpha=1.0)
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
            ls=1                    
            exp_data.align_mean_charge(pred_data, area_between_curves,self._dpH)
            

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
                facecolor="#808080",
                interpolate=True,
                alpha=0.7,
            )
        # Integer labels for y axis
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        # WITH labels on y axis, but indicate the integer values with ticks
        labels = [item.get_text() for item in ax.get_yticklabels()]
        # empty_string_labels = [""] * len(labels)
        # ax.set_yticklabels(empty_string_labels)
        ax.set_ylabel("Mean charge")
        ax.set_xlabel("pH")
        # x-tick every 2 pH units
        ax.set_xticks(np.arange(2.0, 14.0, 2.0))
        # remove top and right spines
        sns.despine()
        # fit everything within bounds
        fig.tight_layout()
        return fig
    
    def plot_predicted_free_energy(
        self, pred_loader: SAMPL6DataProvider) -> matplotlib.figure.Figure:
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
        self, pred_loader: TitrationCurveType,  ) -> matplotlib.figure.Figure:
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
        ax.set_ylim([-0.05,1.05])
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

