import numpy as np
import sys, json

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
        self.second_parquet = kargs['second_parquet'] if kargs['second_parquet'] else print("Second parquet not passed")
        self.racks = kargs['racks'] if kargs['racks'] else print("No racks specified")
        self.nodes = kargs['nodes'] if kargs['nodes'] else print("No nodes specified")
        self.period = kargs['period'] if kargs['period'] else print("No period specified, covid periods taken")


        # Load json file
        with open("/Users/cetinmehmet/Desktop/surfsara-tool/analysis/metric.json", 'r') as f:
            metric_json = json.load(f)

        # Assign the components of the plot
        self.title = metric_json[parquet]['title']
        self.savefig_title = metric_json[parquet]['savefig_title']
        self.ylabel = metric_json[parquet]['ylabel']

        # Get parquet data and load to df
        df = Metric.get_df(parquet, self.node_parquets).replace(-1, np.NaN)
        df.sort_index(inplace=True)

        self.df_dict = {
            # Default period, custom nodes (or racks)
            'df_covid': None,
            'df_non_covid': None,
            'df_rack_covid': None,
            'df_rack_non_covid': None,

            # Custom period (or FULL), custom nodes (or racks)
            'df_custom': None,
            'df_rack': None,
            
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
            self.savefig_title += self.nodes[0] + "_" # proper naming required for saving plots
            df = df.loc[:, self.nodes] # Get nodes

            if self.period is None: # Take full period
                self.df_covid, self.df_non_covid = ParseMetric().covid_non_covid(df) 

                # Adding dfs to lists to prevent "value ambigous error"
                self.df_dict['df_covid'] =  [self.df_covid] 
                self.df_dict['df_non_covid'] = [self.df_non_covid]

            elif self.period == "FULL":
                df_custom = df # Get full period without covid vs non-covid
                self.df_dict['df_custom'] = [df_custom]

            else: # Custom period
                df_custom = ParseMetric().user_period_split(df, self.period[0], self.period[1])

                # Adding dfs to lists to prevent "value ambigous error"
                self.df_dict['df_custom'] = [df_custom]
        
        # Custom racks are specified
        elif self.racks != None and self.nodes == None:
            self.savefig_title += self.racks + "_"
            df_rack = ParseMetric().get_rack_nodes(df, self.racks) # Get rack nodes

            if self.period is None: # Split covid vs non-covid: default analysis
                
                df_rack_covid, df_rack_non_covid = ParseMetric().covid_non_covid(df_rack)

                # Adding dfs to lists to prevent "value ambigous error"
                self.df_dict['df_rack_covid'] =  [df_rack_covid] 
                self.df_dict['df_rack_non_covid'] = [df_rack_non_covid]
                self.savefig_title += "covid_"

            elif self.period == "FULL":
                self.df_dict['df_rack'] = [df_rack]
                self.savefig_title += "full_"

            else: # Custom period
                df_custom = ParseMetric().user_period_split(df_rack, self.period[0], self.period[1])
                self.savefig_title += "period_"
                # Adding dfs to list to prevent "value ambigous error"
                self.df_dict['df_rack'] = [df_custom]


        # Custom nodes or racks aren't specified, so we take the whole node set
        else: 
            self.savefig_title += self.racks + "all_nodes_"

            # Split df to cpu and gpu nodes
            self.df_cpu, self.df_gpu = ParseMetric().cpu_gpu(df)

            if self.period is None: # Take full period: covid vs non-covid
                # Split to df according to covid non covid
                self.df_cpu_covid, self.df_cpu_non_covid = ParseMetric().covid_non_covid(self.df_cpu)
                self.df_gpu_covid, self.df_gpu_non_covid = ParseMetric().covid_non_covid(self.df_gpu)

                # Adding dfs to lists to prevent "value ambigous error"
                self.df_dict.update({
                    'df_cpu_covid': [self.df_cpu_covid], 
                    'df_cpu_non_covid': [self.df_cpu_non_covid], 
                    'df_gpu_covid': [self.df_gpu_covid], 
                    'df_gpu_non_covid': [self.df_gpu_non_covid]})

                self.savefig_title += "covid_"
            
            elif self.period == "FULL": # Get full period without covid vs non-covid
                self.df_dict['df_cpu'] =  [self.df_cpu]
                self.df_dict['df_gpu'] = [self.df_gpu]

                self.savefig_title += "full_"

            else: # Custom period split
                df_cpu = ParseMetric().user_period_split(self.df_cpu, self.period[0], self.period[1])
                df_gpu = ParseMetric().user_period_split(self.df_gpu, self.period[0], self.period[1])
            
                # Adding dfs to lists to prevent "value ambigous error"
                self.df_dict['df_cpu'] =  [df_cpu]
                self.df_dict['df_gpu'] = [df_gpu]

    def get_meta_data(self):
        return self.title, self.savefig_title

    def daily_seasonal_diurnal_pattern(self):
        GraphType(savefig_title = self.savefig_title, 
            title=self.title, period=self.period, ylabel=self.ylabel
        ).custom_daily_seasonal_diurnal_pattern(
            df_dict=self.df_dict, 
        )

    def hourly_seasonal_diurnal_pattern(self):
        GraphType(savefig_title=self.savefig_title, 
            title=self.title, period=self.period, ylabel=self.ylabel
        ).custom_hourly_seasonal_diurnal_pattern(df_dict=self.df_dict)

    def entire_period_analysis(self):
        GraphType().entire_period_analysis(
            df_dict = self.df_dict, 
            ylabel=self.ylabel, 
            title=self.title, 
            savefig_title=self.savefig_title
        )

    # def all_analysis(self):
    #     self.daily_seasonal_diurnal_pattern()
    #     self.daily_monthly_diurnal_pattern()
    #     self.hourly_seasonal_diurnal_pattern()
    #     self.hourly_monthly_diurnal_pattern()
    #     self.rack_analysis()
    #     self.entire_period_analysis()
        

