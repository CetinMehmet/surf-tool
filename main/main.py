import sys, os
import pandas as pd

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
            analyze_metric.cpu.processes_run_block_analysis() # Testing purposes
            pass
        elif command == "gpu":
            """
            Give the user the option to see which analysis (s)he wants
            """
            pass
        elif command == "disk":
            """
            Give the user the option to see which analysis (s)he wants
            """
            pass
        elif command == "memory":
            """
            Give the user the option to see which analysis (s)he wants
            """
            pass
        elif command == "help":
            print("For analysis enter one of the following commands:\ngpu\ncpu\ndisk\nmemory\n")
        else:
            print("Type 'help' for the command menu\n"
                  "Press 'q' to quit")


if __name__ == "__main__":
    main()