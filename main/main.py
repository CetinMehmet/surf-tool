import sys, os, datetime
import pandas as pd
from pathlib import Path
import argparse


sys.path.insert(1, '/Users/cetinmehmet/Desktop/surfsara-tool/parser')
sys.path.insert(2, '/Users/cetinmehmet/Desktop/surfsara-tool/analysis')

from parse_parquet import ParseParquet
from parse_argument import ParseArgument
from analyze_metrics import Metric

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
    node_parquets, gpu_parquets = ParseParquet(dataset_path).get_parquets()
    metric = Metric(node_parquets, gpu_parquets)


    if args.sourcename == "cpu":
        print("Please wait, as we are analyzing...")
        
    elif args.sourcename == "gpu":
        print("Please wait, as we are analyzing...")
        
    elif args.sourcename == "disk":
        print("Please wait, as we are analyzing...")
    
    elif args.sourcename == "memory":
        print("Please wait, as we are analyzing %s..." % (args.sourcename))
        metric.cpu.nr_procs_running.daily_seasonal_diurnal_pattern()
        print('done')

    exit(0)


if __name__ == "__main__":
    main()