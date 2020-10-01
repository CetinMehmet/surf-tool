import numpy as np
import sys

sys.path.insert(1, '/Users/cetinmehmet/Desktop/surfsara-tool/parse_metric')
sys.path.insert(2, '/Users/cetinmehmet/Desktop/surfsara-tool/statistics_scripts')
sys.path.insert(3, '/Users/cetinmehmet/Desktop/surfsara-tool/analysis')

from diurnal_analysis import DiurnalAnalysis
from parse_metric import ParseMetric
from graph_type import GraphType
import matplotlib.pyplot as plt 
import pandas as pd


class Disk(object):

    # First parq: bytes read/written, second parq: read/write completed
    def __init__(self, node_parquets, parquet_rw, parquet_comp):
        from analyze_metrics import Metric # Prevents circular error

        self.node_parquets = node_parquets
        
        if "node_disk_bytes_written" != parquet_rw and "node_disk_bytes_read" != parquet_rw:
            print("wrong order")
            exit(1)

        df_comp = Metric.get_df(parquet_comp, self.node_parquets).replace(-1, np.NaN)
        df_rw = Metric.get_df(parquet_rw, self.node_parquets).replace(-1, np.NaN)

        # Divide bytes read/write to read/write completed
        df = df_rw / df_comp
        df = df / (1024 * 1024)

        # Split df to cpu and gpu nodes
        self.df_cpu, self.df_gpu = ParseMetric().cpu_gpu(df)

        # Split to df according to covid non covid
        self.df_cpu_covid, self.df_cpu_non_covid = ParseMetric().covid_non_covid(self.df_cpu)
        self.df_gpu_covid, self.df_gpu_non_covid = ParseMetric().covid_non_covid(self.df_gpu)

        self.title, self.ylabel, self.savefig_title = "", "", ""
        if parquet_rw == "node_disk_bytes_written":
            self.title = "Total MB written per write operation | All nodes"
            self.ylabel = "MB written"
            self.savefig_title = "MB_written_per_operation_"
        elif parquet_rw == "node_disk_bytes_read":
            self.title = "Total MB read per read operation | All nodes"
            self.ylabel = "MB read"
            self.savefig_title = "bytes_read_per_operation_"
        else:
            print("Wrong parquet given")
        
    def daily_seasonal_diurnal_pattern(self, shareX=True):
        self.savefig_title += "daily_seasonal_v1"
        DiurnalAnalysis().daily_seasonal_diurnal_pattern(
            df_cpu_dic={'covid': self.df_cpu_covid, 
                        'non_covid': self.df_cpu_non_covid,
            }, 
            df_gpu_dic={
                'covid': self.df_gpu_covid,
                'non_covid': self.df_gpu_non_covid
            }, 
            shareX=True, title=self.title, ylabel=self.ylabel,
            savefig_title=self.savefig_title
        )

    def hourly_seasonal_diurnal_pattern(self, shareX=True):
        self.savefig_title += "hourly_seasonal_v1"
        DiurnalAnalysis().hourly_seasonal_diurnal_pattern(
            df_cpu_dic={'covid': self.df_cpu_covid, 
                        'non_covid': self.df_cpu_non_covid,
            }, 
            df_gpu_dic={
                'covid': self.df_gpu_covid,
                'non_covid': self.df_gpu_non_covid
            }, 
            ylabel=self.ylabel, shareX=True, title=self.title, 
            savefig_title=self.savefig_title
        )
 
    def rack_analysis(self):
        self.savefig_title += "avg_per_node_per_rack_v1"
        GraphType().figure_rack_analysis(
            df_cpu_dic={
                'covid': self.df_cpu_covid,
                'non_covid': self.df_cpu_non_covid,
            },
            df_gpu_dic={
                'covid': self.df_gpu_covid, 
                'non_covid': self.df_gpu_non_covid
            }, 
            ylabel=self.ylabel, title=self.title, savefig_title=self.savefig_title
        )
        
    def entire_period_analysis(self):
        # Format the index as dd/mm/yyyy
        self.df_cpu.index = pd.to_datetime(self.df_cpu.index, utc=True, unit="s")
        self.df_gpu.index = pd.to_datetime(self.df_gpu.index, utc=True, unit="s")

        # Get the sum of all the nodes
        self.df_cpu = pd.DataFrame(self.df_cpu).aggregate(func=sum, axis=1)
        self.df_gpu = pd.DataFrame(self.df_gpu).aggregate(func=sum, axis=1)

        GraphType().entire_period_analysis(
            df_cpu=self.df_cpu, df_gpu=self.df_gpu, 
            ylabel=self.ylabel, 
            title=self.title, 
            savefig_title="entire_period_" + self.savefig_title
        )

