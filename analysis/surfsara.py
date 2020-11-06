import numpy as np
import sys
import pandas as pd

sys.path.insert(1, '/Users/cetinmehmet/Desktop/surfsara-tool/parse_metric')
sys.path.insert(2, '/Users/cetinmehmet/Desktop/surfsara-tool/statistics_scripts')
sys.path.insert(3, '/Users/cetinmehmet/Desktop/surfsara-tool/analysis')

from diurnal_analysis import DiurnalAnalysis
from parse_metric import ParseMetric
from graph_type import GraphType

class Surfsara(object):

    def __init__(self, node_parquets, parquet):
        from analyze_metrics import Metric # Prevents circular error

        self.node_parquets = node_parquets
    
        # Get parquet data and load to df
        df = Metric.get_df(metric=parquet, parq_dic=self.node_parquets).replace(-1, np.NaN) 

        df.sort_index(inplace=True)


        # Split df to cpu and gpu nodes
        self.df_cpu, self.df_gpu = ParseMetric().cpu_gpu(df)

        # Split to df according to covid non covid
        self.df_cpu_covid, self.df_cpu_non_covid = ParseMetric().covid_non_covid(self.df_cpu)
        self.df_gpu_covid, self.df_gpu_non_covid = ParseMetric().covid_non_covid(self.df_gpu)
            
        self.title = "Power consumption | SURFsara"
        self.ylabel = "Power consumption(watt)"
        self.savefig_title = "surfsara_nodes_power_consumption"


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

    def daily_monthly_diurnal_pattern(self):
        DiurnalAnalysis().daily_monthly_diurnal_pattern(
            month_dic={'Jan': 1, 'Feb': 2, 'Mar': 3},
            df_cpu=self.df_cpu,
            df_gpu=self.df_gpu,
            savefig_title='daily_monthly_' + self.savefig_title, 
            ylabel=self.ylabel, 
            title=self.title
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

    
    def rack_analysis(self): 
        self.title = "Avg " + self.title
        GraphType().figure_rack_analysis(
            df_cpu_dic={
                'covid': self.df_cpu_covid,
                'non_covid': self.df_cpu_non_covid,
            },
            df_gpu_dic={
                'covid': self.df_gpu_covid, 
                'non_covid': self.df_gpu_non_covid
            }, 
            ylabel=self.ylabel, title=self.title, savefig_title="avg_per_node_per_rack_" + self.savefig_title
        )

    def entire_period_analysis(self):
        # Format the index as dd/mm/yyyy
        self.df_cpu.index = pd.to_datetime(self.df_cpu.index, utc=True, unit="s")
        self.df_gpu.index = pd.to_datetime(self.df_gpu.index, utc=True, unit="s")

        # Get the sum of all the nodes
        df_cpu_aggr = pd.DataFrame(self.df_cpu).aggregate(func=sum, axis=1)
        df_gpu_aggr = pd.DataFrame(self.df_gpu).aggregate(func=sum, axis=1)


        GraphType().entire_period_analysis(
            df_cpu=df_cpu_aggr, df_gpu=df_gpu_aggr, 
            ylabel=self.ylabel, 
            title="Total power consumption", 
            savefig_title=self.savefig_title + "_entire_period" 
        )
        
    
            

