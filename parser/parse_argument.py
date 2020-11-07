import datetime
import argparse
import sys

class ParseArgument:
    def __init__(self):
        self.__parser = argparse.ArgumentParser(description="Instrument to analyze the zenodo dataset", prog="surf_analysis")
    
        self.__parser.add_argument("--path",
                            action="store", type=str, dest="path", required=True,
                            help="The path to the dataset")

        self.__parser.add_argument("-n", "--nodes",
                            action="store", type=self.get_nodes, dest="nodenames", default=None, required=False,
                            help="""Please choose nodes from the following racks:\n
                            Generic racks: 
                                'r1899', 'r1898', 'r1897', 
                                'r1896', 'r1903', 'r1902', 'r1128', 
                                'r1134', 'r1133', 'r1132'\n
                            ML racks: 
                                'r1123', 'r1122', 'r1387',
                                'r1386', 'r1385', 'r1384', 
                                'r1391', 'r1390', 'r1389',
                                'r1379'
                                """
                            )
                        
        self.__parser.add_argument("-p", "--period",
                            action="store", type=self.get_datetime, dest="periodname", default="FULL", nargs=1, required=False,
                            help="Select the periods you would like to analyze.")

        self.__parser.add_argument("-s", "--source", 
                            action="store", type=str, dest="sourcename", choices=['cpu', 'gpu', 'memory', 'disk', 'surfsara'], 
                            required=True, 
                            help="Select the node source you would like to analyze.")

        self.__args = self.__parser.parse_args()


    def get_datetime(self, string):
        if string == "FULL" or string == "":
            return string

        dates = string.split(",")
        date_obj_1 = self.__convert_datetime(dates[0])
        date_obj_2 = self.__convert_datetime(dates[1])

        # Find the start and end time
        start_time = min(date_obj_1, date_obj_2)
        end_time = max(date_obj_1, date_obj_2)

        # Reassign 'period' with datetime type
        return [start_time, end_time]


    def get_nodes(self, string):
        if "," in string:
            # Split from commas, remove them and strip the elements to remove whitespace
            nodes = list(map(lambda x : x.strip(), string.split(",")))
        else:
            # Split by whitespace and filter the empty strings from list
            nodes = list(filter(lambda x: x != "", string.split(" ")))
        
        # Accept no more than 6 nodes
        if len(nodes) > 6:
            print("Error: No more than 6 nodes can be investigated at once.")
            exit(1)

        return nodes


    def get_args(self):
        return self.__args

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



    