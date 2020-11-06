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
from generate_dataset_page import GeneratePage


class Disk(object):

    # First parq: bytes read/written, second parq: read/write completed
    def __init__(self, node_parquets, parquet, *args):
        from analyze_metrics import Metric # Prevents circular error

        parquet2 = args[0] if args else None # Pass read/write time from 'tuple'

        self.node_parquets = node_parquets

        df = Metric.get_df(parquet, self.node_parquets).replace(-1, np.NaN)
        df2 = Metric.get_df(parquet2, self.node_parquets).replace(-1, np.NaN) if parquet2 else print("No parquet 2")
 
        df.sort_index(inplace=True)
        df2.sort_index(inplace=True) if df2 else print("No df2")

        df = df / (1024 * 1024 * 1024) # Convert to GB

        self.title, self.ylabel, self.savefig_title = "", "", ""
        if parquet == "Total GB written":
            self.title = "Mean GB written"
            self.ylabel = "GB written"
            self.savefig_title = "disk_bytes_written_"
            if parquet2 == "node_disk_write_time_ms":
                self.title = "GB/s written"
                self.ylabel = "GB/s written"
                self.savefig_title = "disk_bytes_written_per_sec"
                df = df / (df2 * 0.001) # Divide bytes by seconds

        elif parquet == "node_disk_bytes_read":
            self.title = "Total GB read"
            self.ylabel = "GB read"
            self.savefig_title = "disk_read_per_operation_"
            if parquet2 == "node_disk_read_time_ms":
                self.title = "GB/s read"
                self.ylabel = "GB/s read"
                self.savefig_title = "disk_bytes_read_per_sec"
                df = df / (df2 * 0.001) # Divide bytes by seconds
        else:
            print("Wrong parquet given")

        
        # Split df to cpu and gpu nodes
        self.df_cpu, self.df_gpu = ParseMetric().cpu_gpu(df)

        # Split to df according to covid non covid
        self.df_cpu_covid, self.df_cpu_non_covid = ParseMetric().covid_non_covid(self.df_cpu)
        self.df_gpu_covid, self.df_gpu_non_covid = ParseMetric().covid_non_covid(self.df_gpu)
        
        # self.generate_page = GeneratePage("disk")

    def daily_seasonal_diurnal_pattern(self, shareX=True):
        DiurnalAnalysis().daily_seasonal_diurnal_pattern(
            df_cpu_dic={'covid': self.df_cpu_covid, 
                        'non_covid': self.df_cpu_non_covid,
            }, 
            df_gpu_dic={
                'covid': self.df_gpu_covid,
                'non_covid': self.df_gpu_non_covid
            }, 
            shareX=True, title=self.title, ylabel=self.ylabel,
            savefig_title="daily_seasonal_" + self.savefig_title
        )

        # self.generate_page.launch(self.title, 'daily_seasonal_' + self.savefig_title)
    
    def daily_monthly_diurnal_pattern(self):
        DiurnalAnalysis().daily_monthly_diurnal_pattern(
            month_dic={'Jan': 1, 'Feb': 2, 'Mar': 3},
            df_cpu=self.df_cpu,
            df_gpu=self.df_gpu,
            savefig_title="daily_monthly_" + self.savefig_title, 
            ylabel=self.ylabel, 
            title=self.title
        )

    def hourly_seasonal_diurnal_pattern(self, shareX=True):
        DiurnalAnalysis().hourly_seasonal_diurnal_pattern(
            df_cpu_dic={'covid': self.df_cpu_covid, 
                        'non_covid': self.df_cpu_non_covid,
            }, 
            df_gpu_dic={
                'covid': self.df_gpu_covid,
                'non_covid': self.df_gpu_non_covid
            }, 
            ylabel=self.ylabel, shareX=True, title=self.title, 
            savefig_title="hourly_seasonal_"+self.savefig_title
        )
    
    def hourly_monthly_diurnal_pattern(self):
        DiurnalAnalysis().hourly_monthly_diurnal_pattern(
            month_dic={'Jan': 1, 'Feb': 2, 'Mar': 3},
            df_cpu=self.df_cpu,
            df_gpu=self.df_gpu,
            savefig_title="hourly_monthly_" + self.savefig_title, 
            ylabel=self.ylabel, 
            title=self.title
        )

    def rack_analysis(self):
        GraphType().figure_rack_analysis(
            df_cpu_dic={
                'covid': self.df_cpu_covid,
                'non_covid': self.df_cpu_non_covid,
            },
            df_gpu_dic={
                'covid': self.df_gpu_covid, 
                'non_covid': self.df_gpu_non_covid
            }, 
            ylabel=self.ylabel, title=self.title, savefig_title="rack_analysis_" + self.savefig_title
        )
        
    def entire_period_analysis(self):
        # Format the index as dd/mm/yyyy
        self.df_cpu.index = pd.to_datetime(self.df_cpu.index, utc=True, unit="s")
        self.df_gpu.index = pd.to_datetime(self.df_gpu.index, utc=True, unit="s")

        # Get the sum of all the nodes
        df_cpu_mean = self.df_cpu.mean(axis=1)
        df_gpu_mean = self.df_gpu.mean(axis=1)

        GraphType().entire_period_analysis(
            df_cpu=df_cpu_mean, df_gpu=df_gpu_mean, 
            ylabel=self.ylabel, 
            title="Mean " + self.title, 
            savefig_title="entire_period_mean_" + self.savefig_title
        )

        # self.generate_page.launch(self.title, 'entire_period_' + self.savefig_title)

    """
    # TODO unfinished function
    def correlation_analysis(self):
        df_cpu_covid_sum = self.df_cpu_covid.aggregate(func=sum, axis=1)
        df_gpu_covid_sum = self.df_gpu_covid.aggregate(func=sum, axis=1)
        GraphType().scatter_plot(title="Bytes read/written per operation | pearson = "+ str(GraphType().get_pearsonr(x=self)[0]),
        x=df_cpu_covid_sum.values, y=df_gpu_covid_sum.values)
    """