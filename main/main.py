import sys, os
import pandas as pd
from pathlib import Path

sys.path.insert(1, '/Users/cetinmehmet/Desktop/surfsara-tool/parser')
sys.path.insert(2, '/Users/cetinmehmet/Desktop/surfsara-tool/analysis')

from parse_data import Parser
from analyze_metrics import AnalyzeMetrics


#TODO: find a better place to carry this function
def get_dataset_path(path):
    if path[-1] != '/':
        return path + '/'
    return path


# Get the command line arguments and pass them for thorough analysis
def main():
    if len(sys.argv) != 2:
        print("Please give the folder path to the dataset you would like to analyze"
              "\nusage: <input-path> main.py")
        sys.exit(0)

    dataset_path = get_dataset_path(sys.argv[1])
    node_parquets, gpu_parquets = Parser(dataset_path).parse_data()
    analyze_metric = AnalyzeMetrics(node_parquets, gpu_parquets)

    print("For the type of analysis, please enter one of the following commands:\ndisk\ngpu\nmemory\ncpu")
    command = ""
    while True:
        command = str(input("> ")).strip()
        if command == "q" or command == "exit":
            """
            Exit code when q is pressed or exit is typed
            """
            sys.exit(0)
        elif command == "cpu":
            """
            Give the user the option to see which analysis (s)he wants
            """
            print("Please wait, as we are analyzing...")
            analyze_metric.cpu.nr_procs_running_blocked_analysis() # Testing purposes
            pass
        elif command == "gpu":
            """
            Give the user the option to see which analysis (s)he wants
            """
            print("Please wait, as we are analyzing...")
            pass
        elif command == "disk":
            """
            Give the user the option to see which analysis (s)he wants
            """
            print("Please wait, as we are analyzing...")
            analyze_metric.disk.read_write_analysis()
            pass
        elif command == "memory":
            """
            Give the user the option to see which analysis (s)he wants
            """
            print("Please wait, as we are analyzing...")
            pass
        elif command == "help":
            print("For analysis enter one of the following commands:\ngpu\ncpu\ndisk\nmemory\n")
        else:
            print("Type 'help' for the command menu\n"
                  "Press 'q' to quit")


if __name__ == "__main__":
    main()