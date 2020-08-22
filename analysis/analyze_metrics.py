import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, sys, pyarrow
import seaborn as sns
from pathlib import Path
import calendar


sys.path.insert(1, '/Users/cetinmehmet/Desktop/surfsara-tool/statistics_scripts')
from diurnal_analysis import DiurnalAnalysis


"""
    Analyze the following metrics:
        1. # of processes running-blocked
        2. file description allocation
        3. caches, 
        4. utilized main memory (RAM), 
        5. disk IO, read, and write time
        6. disk IO size [B] + breakdown per read/write,

    Split the metrics in terms of:
        1. Plot the entire set of nodes, which means no partition in nodes
        2. Split the nodes, CPU vs GPU
        3. Break down number of cores for both CPU and GPU
        4. Split from types of processors, there are 5 to 10 types.

    Summarize data:
        1. counts
        2. basic statistics (mean, min, median, max, other quartiles, stddev, CoV if possible)
        3. meta-metric

    Diurnal Analysis:
        1. Hourly analysis:
            a. "Aggregated" over the entire period
            b. "Aggregated" per month 

            The hourly analysis allows us to see differences in office hours (9-5) vs non-office hours; 
            Did the covid period affect people’s working habits (did they do more outside office hours for example?)
        2. Daily analysis:
            a. "Aggregated" over the entire period
            b. "Aggregate" per month

        3. Workday vs weekend:
            a. "Aggregated" over the entire period
            b. "Aggregate" per month
        
        4. Monthly (seasonal patterns):
            a. aggregate all data per month (for each metric → one value per month, 
            or the basic statistics (mean, min, median, max, other quartiles, stddev, CoV if possible))

        5. Per node per metric create a plot to inspect per node.
        6. Inspect Eigen values per node/metric.


"""

TOOL_PATH = Path(os.path.abspath(__file__)).parent.parent # Getting the parent of the parent of the current files path

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

    @staticmethod
    def get_converted_xticks(ax):
        """
        :param ax:
        :return list of day and month strings
        """
        return [pd.to_datetime(tick, unit='d').date().strftime("%d\n%b") for tick in ax.get_xticks()]


class Disk:

    def __init__(self, node_parquets):
        self.node_parquets = node_parquets

    def IO_time_analysis(self):
        pass

    def read_write_time_analysis(self):
        """
        Function can be split to two if it grows big
        """
        pass

    def read_written_bytes_analysis(self):
        pass

    def read_write_completed_analysis(self):
        pass



class Cpu:

    def __init__(self, node_parquets):
        self.node_parquets = node_parquets

    # Perfom diurnal analysis

    def nr_procs_running_blocked_analysis(self):
        df_run = AnalyzeMetrics.get_df("node_procs_running", self.node_parquets).replace(-1, np.NaN)
        df_block = AnalyzeMetrics.get_df("node_procs_blocked", self.node_parquets).replace(-1, np.NaN)
        pass

class Memory:

    def __init__(self, node_parquets):
        self.node_parquets = node_parquets

    def file_descriptor_analysis(self):
        df_alloc = AnalyzeMetrics.get_df("node_filefd_allocated", self.node_parquets)
        df_max = AnalyzeMetrics.get_df("node_filefd_maximum", self.node_parquets)
        pass

    def cache_analysis(self):
        df = AnalyzeMetrics.get_df("node_memory_Buffers", self.node_parquets)
        pass

    def buffer_analysis(self):
        df = AnalyzeMetrics.get_df("node_memory_Cached", self.node_parquets)
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

