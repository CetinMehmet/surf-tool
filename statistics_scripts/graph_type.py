from scipy.stats.stats import pearsonr, spearmanr, kendalltau
import matplotlib.pyplot as plt
import sys, os
from pathlib import Path
import matplotlib.pylab as pylab
import numpy as np
import pandas as pd
import scipy

sys.path.insert(1, '/Users/cetinmehmet/Desktop/surfsara-tool/statistics_scripts')
sys.path.insert(2, '/Users/cetinmehmet/Desktop/surfsara-tool/parser')
sys.path.insert(3, '/Users/cetinmehmet/Desktop/surfsara-tool/analysis')

from parse_metric import ParseMetric


"""
    Since color blinds may also be using the tool, no red or green colors shall be used for any plot
"""



"""
    Spearman:
        - Is there a statistically significant relationship between participants
        - The assumptions of the Spearman correlation are that data must be at least ordinal,
        and the scores on one variable must be monotonically related to the other variable.

    Pearson:
        - Is used to measure the degree of the relationship between linearly related variables.
        - Is there a statistically significant relationship.
        - For the Pearson r correlation, both variables should be normally distributed,
        (normally distributed variables have a bell-shaped curve).
        Other assumptions include linearity and homoscedasticity.
        Linearity assumes a straight line relationship between each of the two variables,
        and homoscedasticity assumes that data is equally distributed about the regression line.

    Kendall:
        - is a non-parametric test that measures the strength of dependence between two variables.
"""

DAY = 24
MID_DAY = int(DAY / 2)
WEEK = 7 * DAY
TOOL_PATH = Path(os.path.abspath(__file__)).parent.parent
MARKERS = ['s', '*', 'o', 'v', '<', 'p', '.', 'd']
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    

# Configure label sizes of graphs
params = {
    'xtick.labelsize':16,
    'ytick.labelsize':16,
    'axes.labelsize':18,
    'figure.figsize': (10, 8),
    'savefig.format': 'pdf',
    'axes.titlesize': 20,
    'legend.loc': 'best',
    'legend.fontsize': "large"
}

pylab.rcParams.update(params)


class GraphType:
    def __init__(self):
        from diurnal_analysis import DiurnalAnalysis
        self.diurnal_analysis = DiurnalAnalysis()

    def __get_converted_xticks(self, ax):
        """
        :param ax:
        :return list of day strings
        """
        return [pd.to_datetime(tick, unit='d').date().strftime("%d\n%b") for tick in ax.get_xticks()]

    def __axes_hourly_plot(self, ax, df_covid, df_non_covid, title, ylabel, xlabel=None):
        ax.plot(df_covid, marker=".", label="covid")
        ax.plot(df_non_covid, marker="*", label="non-covid")
        ax.set_ylim(0, )
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left')

        return ax

    def __axes_daily_seasonal_plot(self, ax, df_covid, df_non_covid, title, ylabel, xlabel=None):
        ax.plot(df_covid, marker=".", label="covid")
        ax.plot(df_non_covid, marker="*", label="non-covid")
        ax.set_ylim(0, )
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
        
        xcoords = [0] + [xcoord for xcoord in range(23, WEEK, DAY)]
        for xc in xcoords:
            ax.axvline(x=xc, color="gray", lw=0.5)
        
        return ax
       

    # This function belongs to Laurens Versluis: https://github.com/lfdversluis
    def __axes_rack_analysis(self, ax, df_covid, df_non_covid, xlabel=None, ylabel=None, title=None):
        rack_nodes = self.__get_rack_nodes(df_covid) #To get the rack nodes
        index=0
        w = 0.4
        ax1, ax2 = plt.axes, plt.axes
        for rack, columns in rack_nodes.items():
            arr_covid = df_covid[list(columns)].values.ravel()
            arr_non_covid = df_non_covid[list(columns)].values.ravel()
            arr_covid = arr_covid[arr_covid >= 0]
            arr_non_covid = arr_non_covid[arr_non_covid >= 0] # Filter all non-zero values

            ax1 = ax.bar(x=index - w/2, height=arr_covid.mean(), width=w, yerr=arr_covid.std(), color="blue", capsize=5)
            ax2 = ax.bar(x=index + w/2, height=arr_non_covid.mean(), width=w, yerr=arr_non_covid.std(), color="red", capsize=5)
            index += 1

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(handles=[ax1, ax2], labels=['covid', 'non-covid'], loc="upper right")
        ax.set_xticks(np.arange(len(rack_nodes.keys())))
        ax.set_xticklabels(rack_nodes.keys(), rotation = 90)

    def __get_rack_nodes(self, df):
        rack_nodes = {}

        for node in df.columns:
            rack = node.split("n")[0]
            if rack not in rack_nodes:
                rack_nodes[rack] = set()

            rack_nodes[rack].add(node)

        return rack_nodes

    def figure_daily_per_seasonal(
        self, df_cpu_dic, df_gpu_dic, shareX=True, xlabel=None, ylabel=None, title_cpu=None, title_gpu=None, savefig_title=None
    ):

        fig, (ax_cpu, ax_gpu) = plt.subplots(2, 1, sharex=shareX, constrained_layout=True)

        self.__axes_daily_seasonal_plot(
            ax=ax_cpu, 
            df_covid=df_cpu_dic["covid"], 
            df_non_covid=df_cpu_dic["non_covid"], 
            ylabel=ylabel,
            title=title_cpu + " | aggregated full period"
        )

        ax_gpu = self.__axes_daily_seasonal_plot(
            ax=ax_gpu, 
            df_covid=df_gpu_dic["covid"], 
            df_non_covid=df_gpu_dic["non_covid"], 
            ylabel=ylabel,
            title=title_gpu + " | aggregated full period",
            xlabel=xlabel
        )
        ax_gpu.set_xticks([tick for tick in range(MID_DAY-1, WEEK, DAY)])
        ax_gpu.set_xticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])

        plt.savefig(os.path.join(str(TOOL_PATH) + "/plots/", savefig_title + ".pdf"), dpi=100) 
        plt.show()
        plt.pause(0.0001)


    def __construct_daily_montly_plots(self, ax, title=None, ylabel=None):
        ax.set_ylim(0, )
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left')
        
        xcoords = [0] + [xcoord for xcoord in range(23, WEEK, DAY)]
        for xc in xcoords:
            ax.axvline(x=xc, color="gray", lw=0.5)

    def __construct_hourly_montly_plots(self, ax, ylabel, title):
        ax.set_ylim(0, )
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left')

    def figure_daily_per_monthly(self, df_cpu, df_gpu, month_dic, savefig_title, ylabel, title):

        fig, (ax_cpu, ax_gpu) = plt.subplots(2, 1, sharex=True, constrained_layout=True)

        for name, value in month_dic.items():
            df_cpu_month = self.diurnal_analysis.get_daily_month_df(df_cpu, value)
            df_gpu_month = self.diurnal_analysis.get_daily_month_df(df_gpu, value)

            ax_cpu.plot(df_cpu_month, marker=MARKERS[value], label=name, color=COLORS[value])
            ax_gpu.plot(df_gpu_month, marker=MARKERS[value], label=name, color=COLORS[value])

        # After plotting the lines, now construct the graph
        self.__construct_daily_montly_plots(ax=ax_cpu, ylabel=ylabel, title = title + " | CPU nodes | aggregated per month")
        self.__construct_daily_montly_plots(ax=ax_gpu, ylabel=ylabel, title = title + " | GPU nodes | aggregated per month")


        ax_gpu.set_xticks([tick for tick in range(MID_DAY-1, WEEK, DAY)])
        ax_gpu.set_xticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])

        plt.savefig(os.path.join(str(TOOL_PATH) + "/plots/", savefig_title + ".pdf"), dpi=100) 
        plt.show()
        plt.pause(0.0001)

    def figure_hourly_monthly(self, df_cpu, df_gpu, month_dic, savefig_title, ylabel, title):
        fig, (ax_cpu, ax_gpu) = plt.subplots(2, 1, sharex=True, constrained_layout=True)

        for name, value in month_dic.items():
            df_cpu_month = self.diurnal_analysis.get_hourly_month_df(df_cpu, value)
            df_gpu_month = self.diurnal_analysis.get_hourly_month_df(df_gpu, value)

            ax_cpu.plot(df_cpu_month, marker=MARKERS[value], label=name, color=COLORS[value])
            ax_gpu.plot(df_gpu_month, marker=MARKERS[value], label=name, color=COLORS[value])

        # After plotting the lines, now construct the graph
        self.__construct_hourly_montly_plots(ax=ax_cpu, ylabel=ylabel, title = title + " | CPU nodes | aggregated per month")
        self.__construct_hourly_montly_plots(ax=ax_gpu, ylabel=ylabel, title = title + " | GPU nodes | aggregated per month")

        plt.savefig(os.path.join(str(TOOL_PATH) + "/plots/", savefig_title + ".pdf"), dpi=100) 
        plt.show()
        plt.pause(0.0001)


    def figure_hourly_seasonal(
        self, df_cpu_dic, df_gpu_dic, ylabel=None, 
        shareX=None, title_cpu=None, title_gpu=None, savefig_title=None
    ):
        fig, (ax_cpu, ax_gpu) = plt.subplots(2, 1, sharex=shareX, constrained_layout=True)

        self.__axes_hourly_plot(
            ax=ax_cpu, 
            df_covid=df_cpu_dic["covid"], 
            df_non_covid=df_cpu_dic["non_covid"], 
            ylabel=ylabel,
            title=title_cpu
        )

        self.ax_gpu = self.__axes_hourly_plot(
            ax=ax_gpu, 
            df_covid=df_gpu_dic["covid"], 
            df_non_covid=df_gpu_dic["non_covid"], 
            ylabel=ylabel,
            title=title_gpu,
            xlabel="Hours"
        )
        ax_gpu.set_xticks([hour for hour in range(0, 24, 2)])
        ax_gpu.set_xticklabels([hour for hour in range(0, 24, 2)])

        plt.savefig(os.path.join(str(TOOL_PATH) + "/plots/", savefig_title + ".pdf"), dpi=100) 
        plt.show()
        plt.pause(0.0001)

    def figure_rack_analysis(self, df_cpu_dic, df_gpu_dic, ylabel, title=None, savefig_title=None):

        _, (ax_cpu, ax_gpu) = plt.subplots(2, 1, constrained_layout=True)
        self.__axes_rack_analysis(ax_gpu, xlabel="GPU racks", ylabel=ylabel, df_covid=df_gpu_dic["covid"], df_non_covid=df_gpu_dic["non_covid"], title=title)
        self.__axes_rack_analysis(ax_cpu, xlabel="CPU racks", ylabel=ylabel, df_covid=df_cpu_dic["covid"], df_non_covid=df_cpu_dic["non_covid"], title=title)
        
        plt.savefig(os.path.join(str(TOOL_PATH) + "/plots/", savefig_title + ".pdf"), dpi=100) 
        plt.show()
        plt.pause(0.0001)
    
    def entire_period_analysis(self, df_cpu, df_gpu, ylabel=None, title=None, savefig_title=None):

        _, (ax) = plt.subplots( figsize=(18,10))
        ax.plot(df_cpu, label="cpu", color="blue")
        ax.plot(df_gpu, label="gpu", color="red")
        ax.set_ylim(0, )
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Time")
        ax.set_title(title)
        ax.legend(loc="lower right", fontsize=18)
        ax.set_xticklabels(labels=self.__get_converted_xticks(ax))

        plt.savefig(os.path.join(str(TOOL_PATH) + "/plots/", savefig_title + ".pdf"), dpi=100) 
        plt.show()
        plt.pause(0.0001)

    def scatter_plot(self, title, x, y, savefig_title):
        _, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(x=x, y=y, marker='*')
        ax.set_xlabel("Read", fontsize=16)
        ax.set_ylabel("Write", fontsize=16)
        ax.set_title(title, fontsize=18)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        plt.savefig(os.path.join(str(TOOL_PATH) + "/plots/", savefig_title + ".pdf"), dpi=100) 
        plt.show()
        plt.pause(0.0001)
        

    def get_pearsonr(self, x, y):
        return scipy.stats.pearsonr(x=x, y=y)[0] # Return r which is pearson correlation coefficient


