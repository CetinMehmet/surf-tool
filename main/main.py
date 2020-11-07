import sys, os, datetime
import pandas as pd
from pathlib import Path
import argparse
import time

sys.path.insert(1, '/Users/cetinmehmet/Desktop/surfsara-tool/parser')
sys.path.insert(2, '/Users/cetinmehmet/Desktop/surfsara-tool/analysis')
sys.path.insert(3, '/Users/cetinmehmet/Desktop/surfsara-tool/')

from parse_parquet import ParseParquet
from parse_argument import ParseArgument
from analyze_metrics import Metric
from generate_dataset_page import GeneratePage

"""
analysis --period="FULL_DATASET" --source:"cpu"
"""


"""
    Covid: [27th of February - End of the current dataset]
    NON-COVID: [Start of the current dataset - 26th of February]
"""


# TODO: find a better place to carry this function
def get_dataset_path(path):
    if path[-1] != '/':
        return path + '/'
    return path


# Get the command line arguments and pass them for through analysis
def main():

    args = ParseArgument().get_args() # Get the arguments from the command line

    # Get the dataset path, parse the data to 2 dictionaries containing node and gpu parquet paths
    dataset_path = get_dataset_path(args.path)
    new_node_parquets, node_parquets, gpu_parquets = ParseParquet(dataset_path).get_parquets()
    metric = Metric(new_node_parquets, node_parquets, gpu_parquets)

    # Get start and endtime
    period = args.periodname[0]

    # Get nodes
    nodes = args.nodenames

    # Default analysis
    if period == "" and nodes == []:
        pass


    # Custom analysis
    else:
        pass

    if args.sourcename == "cpu": # Metrics: procs_running, procs_blocked
        print("Please wait, as we are analyzing...")
       
        # metric.custom("node_procs_running", parquet_total=None, nodes=nodes, period=period).entire_period_analysis()
        # metric.custom("node_load1", parquet_total=None, nodes=nodes, period=period).custom_daily_seasonal_diurnal_pattern()
        metric.custom("node_load1", parquet_total=None, nodes=nodes, period=period).custom_hourly_seasonal_diurnal_pattern()
        print("Done!")
       
    elif args.sourcename == "disk": #
        print("Please wait, as we are analyzing %s..." % (args.sourcename))
        
        print("Done!")
    
    elif args.sourcename == "memory":
        print("Please wait, as we are analyzing %s..." % (args.sourcename))
       
        print("Done!")

    elif args.sourcename == "surfsara":
        print("Please wait, as we are analyzing %s..." % (args.sourcename))
        

        print("Done!")

    exit(0)


if __name__ == "__main__":
    main()