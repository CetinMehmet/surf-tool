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


class CustomAnalysis(object):

    def __init__(self, node_parquets, parquet, **kargs):
        from analyze_metrics import Metric # Prevents circular error

        self.node_parquets = node_parquets
        self.parquet = parquet  
        self.parquet_total = kargs['parquet_total'] if kargs['parquet_total'] else print("Second parquet not passed")
        self.nodes = kargs['nodes'] if kargs['nodes'] else print("No nodes specified")
        self.period = kargs['period'] if kargs['period'] else print("No period specified, full taken")

         # Get parquet data and load to df
        df = Metric.get_df(parquet, self.node_parquets).replace(-1, np.NaN)
        df.sort_index(inplace=True)

        self.df_dict = {
            # Default period, custom nodes
            'df_covid': None,
            'df_non_covid': None,

            # Custom period, custom nodes
            'df_custom': None,
            
            # Default period, no nodes specified
            'df_cpu_covid': None,
            'df_gpu_covid': None,
            'df_cpu_non_covid': None,
            'df_gpu_non_covid': None,

            # Custom period, no nodes specified
            'df_cpu': None,
            'df_gpu': None
        }

        if self.nodes != None: 
            df = df.loc[:, self.nodes] # Get nodes

            if self.period is None: # Take full period
                self.df_covid, self.df_non_covid = ParseMetric().covid_non_covid(df) 

                # Adding dfs to lists to prevent "value ambigous error"
                self.df_dict['df_covid'] =  [self.df_covid] 
                self.df_dict['df_non_covid'] = [self.df_non_covid]
         

            else: # Custom period
                self.df_custom = ParseMetric().user_period_split(df, self.period[0], self.period[1])

                # Adding dfs to lists to prevent "value ambigous error"
                self.df_dict['df_custom'] = [self.df_custom]
        
        # Custom nodes aren't specified, so we take the whole node set
        else: 
            # Split df to cpu and gpu nodes
            self.df_cpu, self.df_gpu = ParseMetric().cpu_gpu(df)

            if self.period is None: # Take full period
                # Split to df according to covid non covid
                self.df_cpu_covid, self.df_cpu_non_covid = ParseMetric().covid_non_covid(self.df_cpu)
                self.df_gpu_covid, self.df_gpu_non_covid = ParseMetric().covid_non_covid(self.df_gpu)

                # Adding dfs to lists to prevent "value ambigous error"
                self.df_dict.update({
                    'df_cpu_covid': [self.df_cpu_covid], 
                    'df_cpu_non_covid': [self.df_cpu_non_covid], 
                    'df_gpu_covid': [self.df_gpu_covid], 
                    'df_gpu_non_covid': [self.df_gpu_non_covid]})

            else: # Custom period split
                self.df_cpu = ParseMetric().user_period_split(self.df_cpu, self.period[0], self.period[1])
                self.df_gpu = ParseMetric().user_period_split(self.df_gpu, self.period[0], self.period[1])
            
                # Adding dfs to lists to prevent "value ambigous error"
                self.df_dict['df_cpu'] =  [self.df_cpu]
                self.df_dict['df_gpu'] = [self.df_gpu]

        self.title, self.ylabel, self.savefig_title = "", "", ""
        if parquet == "node_procs_running":
            self.title = "Total number of running procs"
            self.ylabel = "Running procs"
            self.savefig_title = "cpu_running_procs"

        elif parquet == "node_procs_blocked":
            self.title = "Total number of blocked procs"
            self.ylabel = "Blocked procs"
            self.savefig_title = "cpu_blocked_procs"

        elif parquet == "node_load1":
            self.title = "Load1"
            self.ylabel = "Load1"
            self.savefig_title = "load1"


    def get_meta_data(self):
        return self.title, self.savefig_title


    def custom_daily_seasonal_diurnal_pattern(self):
        GraphType().custom_daily_seasonal_diurnal_pattern(
            df_dict=self.df_dict, 
            savefig_title=self.savefig_title, 
            title=self.title,
            period=self.period,
            ylabel=self.ylabel
        )

    def custom_hourly_seasonal_diurnal_pattern(self):
        GraphType().custom_hourly_seasonal_diurnal_pattern(
            df_dict=self.df_dict, 
            savefig_title=self.savefig_title, 
            title=self.title,
            period=self.period,
            ylabel=self.ylabel
        )

    def entire_period_analysis(self):

        GraphType().entire_period_analysis(
            df_dict = self.df_dict, 
            ylabel=self.ylabel, 
            title=self.title, 
            savefig_title="entire_period_" + self.savefig_title
        )

    # def all_analysis(self):
    #     self.daily_seasonal_diurnal_pattern()
    #     self.daily_monthly_diurnal_pattern()
    #     self.hourly_seasonal_diurnal_pattern()
    #     self.hourly_monthly_diurnal_pattern()
    #     self.rack_analysis()
    #     self.entire_period_analysis()
        

