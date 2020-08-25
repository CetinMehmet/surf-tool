import sys, os, datetime
import pandas as pd
from pathlib import Path
import argparse


sys.path.insert(1, '/Users/cetinmehmet/Desktop/surfsara-tool/parser')
sys.path.insert(2, '/Users/cetinmehmet/Desktop/surfsara-tool/analysis')

from parse_data import ParseData
from parse_argument import ParseArgument
from analyze_metrics import AnalyzeMetrics

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
    node_parquets, gpu_parquets = ParseData(dataset_path).parse_data()
    analyze_metric = AnalyzeMetrics(node_parquets, gpu_parquets)


    if args.sourcename == "cpu":
        print("Please wait, as we are analyzing...")
        
    elif args.sourcename == "gpu":
        print("Please wait, as we are analyzing...")
        
    elif args.sourcename == "disk":
        print("Please wait, as we are analyzing...")
    
    elif args.sourcename == "memory":
        """
        Give the user the option to see which analysis (s)he wants
        """
        print("Please wait, as we are analyzing...")


if __name__ == "__main__":
    main()