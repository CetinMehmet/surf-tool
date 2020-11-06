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


class Cpu(object):

    def __init__(self, node_parquets, parquet):
        from analyze_metrics import Metric # Prevents circular error

        self.node_parquets = node_parquets
        self.parquet = parquet  

        # Get parquet data and load to df
        df = Metric.get_df(parquet, self.node_parquets).replace(-1, np.NaN)
        df.sort_index(inplace=True)

        # Split df to cpu and gpu nodes
        self.df_cpu, self.df_gpu = ParseMetric().cpu_gpu(df)

        # Split to df according to covid non covid
        self.df_cpu_covid, self.df_cpu_non_covid = ParseMetric().covid_non_covid(self.df_cpu)
        self.df_gpu_covid, self.df_gpu_non_covid = ParseMetric().covid_non_covid(self.df_gpu)

        self.title, self.ylabel, self.savefig_title = "", "", ""
        if parquet == "node_procs_running":
            self.title = "Total number of running procs"
            self.ylabel = "Running procs"
            self.savefig_title = "cpu_running_procs"

        elif parquet == "node_procs_blocked":
            self.title = "Total number of blocked procs"
            self.ylabel = "Blocked procs"
            self.savefig_title = "cpu_blocked_procs"

        self.generate_page = GeneratePage("cpu")


    def get_meta_data(self):
        return self.title, self.savefig_title

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
            savefig_title='daily_seasonal_' + self.savefig_title
        )

        self.generate_page.launch(self.title, 'daily_seasonal_' + self.savefig_title, self.parquet)
    
    def daily_monthly_diurnal_pattern(self):
        DiurnalAnalysis().daily_monthly_diurnal_pattern(
            month_dic={'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5},
            df_cpu=self.df_cpu,
            df_gpu=self.df_gpu,
            savefig_title="daily_monthly_" + self.savefig_title, 
            ylabel=self.ylabel, 
            title=self.title
        )

        self.generate_page.launch(self.title, 'daily_monthly_' + self.savefig_title, self.parquet)

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
            savefig_title='hourly_seasonal_' + self.savefig_title
        )

        self.generate_page.launch(self.title, 'hourly_seasonal_' + self.savefig_title)
    
    def hourly_monthly_diurnal_pattern(self):
        DiurnalAnalysis().hourly_monthly_diurnal_pattern(
            month_dic={'Jan': 1, 'Feb': 2, 'Mar': 3},
            df_cpu=self.df_cpu,
            df_gpu=self.df_gpu,
            savefig_title="hourly_monthly_" + self.savefig_title, 
            ylabel=self.ylabel, 
            title=self.title
        )

        self.generate_page.launch(self.title, 'hourly_monthly_' + self.savefig_title, self.parquet)

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
            ylabel=self.ylabel, title=self.title, savefig_title=self.savefig_title
        )

        self.generate_page.launch(self.title, 'rack_analysis_' + self.savefig_title, self.parquet)

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

        self.generate_page.launch(self.title, 'entire_period_' + self.savefig_title, self.parquet)

    def all_analysis(self):
        self.daily_seasonal_diurnal_pattern()
        self.daily_monthly_diurnal_pattern()
        self.hourly_seasonal_diurnal_pattern()
        self.hourly_monthly_diurnal_pattern()
        self.rack_analysis()
        self.entire_period_analysis()
        
