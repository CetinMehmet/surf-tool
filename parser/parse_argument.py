import datetime
import argparse
import sys

class ParseArgument:
    def __init__(self):
        self.__parser = argparse.ArgumentParser(description="Instrument to analyze the zenodo dataset", prog="surf_analysis")
    
        self.__parser.add_argument("--path",
                            action="store", type=str, dest="path", required=True,
                            help="The path to the dataset")

        self.__parser.add_argument("--node",
                            action="store", type=str, dest="nodename", choices=["cpu", "gpu"], required=True,
                            help="Select the type of node you would like to analyze.")
                        
        self.__parser.add_argument("-p", "--period",
                            action="store", type=str, dest="period", default="FULL", nargs='*',
                            help="Select the periods you would like to analyze.")

        self.__parser.add_argument("-s", "--source", 
                            action="store", type=str, dest="sourcename", choices=['cpu', 'gpu', 'memory', 'disk'], required=True, 
                            help="Select the node source you would like to analyze.")

        self.__args = self.__parser.parse_args()

        if len(self.__args.period) > 2:
            print("periods can't be more than 2 arguments")
            sys.exit(1)

        # Convert str to seperate dates in the format of YYYY-MM-DD
        elif len(self.__args.period) == 2:
            date_obj_1 = self.__convert_datetime(self.__args.period[0])
            date_obj_2 = self.__convert_datetime(self.__args.period[1])

            # Find the start and end time
            start_time = min(date_obj_1, date_obj_2)
            end_time = max(date_obj_1, date_obj_2)

            # Reassign 'period' with datetime type
            self.__args.period[0] = start_time
            self.__args.period[1] = end_time

        # If it is COVID, FULL, or NON-COVID
        elif len(self.__args.period) == 1:
            self.__modify_choices(self.__parser, "period", ["FULL", "COVID", "NON-COVID"])

        else:
            print("Warning: default period is the 'FULL' period")

    def __modify_choices(self, parser, dest, choices):
        for action in parser._actions:
            if action.dest == dest:
                action.choices = choices
                return
        else:
            raise AssertionError('argument {} not found'.format(dest))

    # Get the periods and convert to datetime object
    def __convert_datetime(self, string):
        try:
            date_obj = datetime.datetime.strptime(string, '%Y-%m-%d')
            date_obj += datetime.timedelta(seconds=0, minutes=0, hours=0)
            return date_obj

        except ValueError:
            print("Date must be formatted as 'yyyy-mm-dd'.")
            sys.exit(1)

    def get_args(self):
        return self.__args


    