import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, sys, pyarrow
import seaborn as sns
from pathlib import Path


sys.path.insert(1, '/Users/cetinmehmet/Desktop/surfsara-tool/statistics_scripts')
from diurnal_analysis import DiurnalAnalysis


"""
    Analyze metrics individually and plot them to a directory
"""

TOOL_PATH = Path(os.path.abspath(__file__)).parent.parent

class AnalyzeMetrics:

    def __init__(self, node_parquets: dict, gpu_parquets: dict):
        self.node_parquets = node_parquets
        self.gpu_parquets = gpu_parquets

        self.disk = Disk(self.node_parquets)
        self.memory = Memory(self.node_parquets)
        self.cpu = Cpu(self.node_parquets)
        self.gpu = Gpu(self.gpu_parquets)

    @staticmethod
    def __get_parquet_path(metric, parq_dic):
        for key, value in parq_dic.items():
            if key == metric:
                return value

    @staticmethod
    def get_df(metric, parq_dic):
        """
        return the df for the corresponding "metric" from the "parquet dict"
        :param metric:
        :param parq_dic:
        :return:
        """
        path = AnalyzeMetrics.__get_parquet_path(metric, parq_dic)
        return pd.read_parquet(path, engine="pyarrow")


class Disk:

    def __init__(self, node_parquets):
        self.node_parquets = node_parquets

    def entropy_availability_analysis(self):
        """
        Plot histogram
        Diurnal analysis
        Normalized lineplot
        :return:
        """
        df = AnalyzeMetrics.get_df("node_entropy_available_bits", self.node_parquets)

    #TODO: Fix this function
    def read_write_analysis(self):
        """
        Normalized line plot
        Line plot every read/write size
        :return:
        """

        # Read parquets to df and get the mean of the nodes
        df_read_bytes = AnalyzeMetrics.get_df("node_disk_bytes_read", self.node_parquets).mean(axis=1)
        df_write_bytes = AnalyzeMetrics.get_df("node_disk_bytes_written", self.node_parquets).mean(axis=1)
        df_read_completed = AnalyzeMetrics.get_df("node_disk_reads_completed", self.node_parquets).mean(axis=1)
        df_write_completed = AnalyzeMetrics.get_df("node_disk_writes_completed", self.node_parquets).mean(axis=1)

        # Adjust the seconds to "date"
        df_read_bytes.index = pd.to_datetime(df_read_bytes.index, unit='s')
        df_write_bytes.index = pd.to_datetime(df_write_bytes.index, unit='s')
        df_read_completed.index = pd.to_datetime(df_read_completed.index, unit='s')
        df_write_completed.index = pd.to_datetime(df_write_completed.index, unit='s')

        df_readsize = df_read_bytes/df_read_completed
        df_writesize = df_write_bytes/df_write_completed

        # Create subplots for line plots
        fig, (ax_read, ax_write) = plt.subplots(2, 1, figsize=(8, 10), sharex=True)
        fig.tight_layout(pad=3.0) 

        # Line plot for write chunk size for each disk operation
        ax_read.plot(df_readsize, color="blue", label="read size")
        ax_read.set_title("Read bytes per operation")
        ax_read.set_ylabel("Bytes")

        ax_write.plot(df_writesize, color="red", label="write size")
        ax_write.set_title("Written bytes per operation")
        ax_write.set_ylabel("Bytes")

        ax_write.tick_params(axis='x', labelrotation=0, labelsize=8)

        plt.savefig(os.path.join(str(TOOL_PATH) + "/plots/", "read_write_analysis.pdf"), dpi=100)
        plt.show(block=False) 
        plt.pause(0.001) # Enables the program to keep running after the plot is displayed



class Cpu:

    def __init__(self, node_parquets):
        self.node_parquets = node_parquets

    def nr_procs_running_blocked_analysis(self):
        """
        Normalized lineplot
        """
        df_run = AnalyzeMetrics.get_df("node_procs_running", self.node_parquets).replace(-1, np.NaN)
        df_block = AnalyzeMetrics.get_df("node_procs_blocked", self.node_parquets).replace(-1, np.NaN)

        df_run = DiurnalAnalysis.get_diurnal_df(df_run)
        df_block = DiurnalAnalysis.get_diurnal_df(df_block)

        df_run_dt = df_run.groupby("dt").mean()
        df_run_mean = df_run_dt.mean(axis=1) # get mean of nodes

        df_block_dt = df_block.groupby("dt").mean()
        df_block_mean = df_block_dt.mean(axis=1) # get mean of nodes

        df_run_mean_normalized = df_run_mean / max(df_run_mean.values)
        df_block_mean_normalized = df_block_mean / max(df_block_mean.values)

        # Create subplots for cdf and normalized plots
        fig, (ax_line, ax_cdf) = plt.subplots(2, 1, figsize=(10, 10))
        fig.tight_layout(pad=3.0) # Increase distance between plots
        # fig.suptitle("CDF and Normalized Line Plot for # Processes") # Title of the figure

        ax_line.plot(df_run_mean_normalized, color="blue", label="running")
        ax_line.plot(df_block_mean_normalized, color="red", label="blocked")
        ax_line.set_xlabel("Time")
        ax_line.set_title("# processes")
        ax_line.tick_params(axis='x', labelrotation=0, labelsize=8)
        ax_line.legend(loc="upper right")

        ax_cdf.hist(df_run_mean, bins=100, density=True, cumulative=True, histtype="step", color="red", label="run")
        ax_cdf.hist(df_block_mean, bins=100, cumulative=True, density=True, histtype="step", label="block")
        ax_cdf.set_title("CDF of # process")
        ax_cdf.legend(loc="center")
        ax_cdf.grid(True)
        ax_cdf.set_xlabel("# Process")
        ax_cdf.set_ylabel("Density")

        plt.savefig(os.path.join(str(TOOL_PATH) + "/plots/", "procs_run_block_analysis.pdf"), dpi=100)
        plt.show(block=False)
        plt.pause(0.001) # This enables the program to keep running after the plot is displayed

    def load_diurnal_analysis(self):
        """
        plot diurnal graphs for load1, load5, load15
        :return:
        """
        df_load1 = AnalyzeMetrics.get_df("node_load1", self.node_parquets)
        df_load5 = AnalyzeMetrics.get_df("node_load5", self.node_parquets)
        df_load15 = AnalyzeMetrics.get_df("node_load15", self.node_parquets)

        loads_df = DiurnalAnalysis(df1, df5, df15)
        loads_df.plot_analysis("hour", "hourly load analysis")


class Memory:

    def __init__(self, node_parquets):
        self.node_parquets = node_parquets

    def file_descriptor_analysis(self):
        pass

    def commit_analysis(self):
        pass


class Gpu:

    def __init__(self, gpu_parquets):
        self.gpu_parquets = gpu_parquets

    # Written by: Lauren Versluis
    def __unpack_bits(self, df):
        def parse_node_row(row, index):
            def unpack_bitmask(val):
                return [np.bitwise_and(np.right_shift(val, i * 16), 0xFFFF) for i in range(4)]

            # Run unpack_bitmask over the array, creating an array of arrays, each sub array having 4 elements
            # the unpacked values. Then, concatenate them all into a list and flatten it using ravel.
            arr = np.concatenate(np.apply_along_axis(unpack_bitmask, 0, row)).ravel()
            # Now that we have a list of all individual wattages, create a series and give it the multi-index
            # so that it knows to iteratively match it
            return pd.Series(arr, index=index)

        # Construct a multi-index, for each node (a column in the old dataframe),
        # we create another index, one per GPU (4 gpus per node).
        iterables = [df.columns, ["gpu{}".format(i) for i in range(4)]]
        midx = pd.MultiIndex.from_product(iterables, names=['node', 'gpu'])

        # Parse row by row (thus along the 'columns' axis), and pass the multi-index so that we can create a new multi-index dataframe.
        df = df.apply(parse_node_row, axis="columns", result_type="expand", args=(midx,))

        return df

    def fanspeed_temperature_analysis(self):
        """
        Provides line, correlation, and diurnal plots
        :return:
        """
        fanspeed_df = AnalyzeMetrics.get_df("gpu_fanspeed_percent", self.gpu_parquets)
        temperature_df = AnalyzeMetrics.get_df("gpu_temperature_celcius", self.gpu_parquets)

