import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, sys, pyarrow
import seaborn as sns
from pathlib import Path
import calendar


sys.path.insert(1, '/Users/cetinmehmet/Desktop/surfsara-tool/statistics_scripts')
sys.path.insert(2, '/Users/cetinmehmet/Desktop/surfsara-tool/parse_metric')
sys.path.insert(3, '/Users/cetinmehmet/Desktop/surfsara-tool/analysis')


from diurnal_analysis import DiurnalAnalysis
from parse_metric import ParseMetric
from graph_type import GraphType
from cpu import Cpu
from memory import Memory
from disk import Disk
from surfsara import Surfsara
from custom_analysis import CustomAnalysis

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

TOOL_PATH = Path(os.path.abspath(__file__)).parent.parent


class Metric:

    def __init__(self, new_node_parquets: dict, node_parquets: dict, gpu_parquets: dict):
        self.node_parquets = node_parquets
        self.gpu_parquets = gpu_parquets
        self.new_node_parquets = new_node_parquets

    def cpu(self, parquet, **kargs):
        parquet_total = kargs['parquet_total']
        nodes = kargs['nodes']
        periods = kargs['periods']

        return Cpu(node_parquets=self.new_node_parquets, parquet=parquet, parquet_total=parquet_total, nodes=nodes, periods=periods) # New dataset is tested
    
    def custom(self, parquet, **kargs):
        parquet_total = kargs['parquet_total']
        nodes = kargs['nodes']
        period = kargs['period']

        return CustomAnalysis(node_parquets=self.new_node_parquets, parquet=parquet, parquet_total=parquet_total, nodes=nodes, period=period) # New dataset is tested

    def disk(self, parquet, *args):
        if args:
            return Disk(self.node_parquets, parquet, args[0]) # second metric
        else:
            return Disk(self.node_parquets, parquet)

    def memory(self, parquet, *args):
        return Memory(self.node_parquets, parquet, args[0])

    def surfsara(self, parquet):
        return Surfsara(self.node_parquets, parquet)

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
        path = Metric.__get_parquet_path(metric, parq_dic)
        return pd.read_parquet(path, engine="pyarrow")

    def construct_table(self, df):
        df_table = self.__get_table_df(df)
        fig, ax = plt.subplots()

        # hide axes
        fig.patch.set_visible(False)
        fig.tight_layout()

        ax.axis('off')
        ax.axis('tight')

        row_labels = ['mean', 'min', 'median', 'max', '1st quartile', '3rd quartile', 'standard deviation']
        ax.table(
            cellText=df_table.values,
            colLabels=df_table.columns,
            rowLabels=row_labels,
            colWidths=[.4 for i in range(len(row_labels))],
            loc='center'
        )

        plt.savefig(os.path.join(str(TOOL_PATH) + "/plots/", "table" + ".pdf"), dpi=100)

    def __get_table_df(self, df):
        df_table = pd.DataFrame(
            data={'metric_df' : {
                'mean': df.mean(),
                'min' : df.min(),
                'median': df.median(),
                'max': df.max(),
                '1st quartile': df.quantile(.25),
                '3rd quartile': df.quantile(.75),
                'Standard deviation': df.std()
            }}
        )

        return df_table








