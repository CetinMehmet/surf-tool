import pandas as pd

"""
    Split the metrics in terms of:
        1. Plot the entire set of nodes, which means no partition in nodes
        2. Split the nodes, CPU vs GPU
        3. Break down number of cores for both CPU and GPU
        4. Split from types of processors, there are 5 to 10 types.
        5. COVID vs NON-COVID
"""

# The CPU racks
CPU_RACKS = [
    'r1899', 'r1898', 'r1897', 
    'r1896', 'r1903', 'r1902', 'r1128', 
    'r1134', 'r1133', 'r1132'
]

class ParseMetric:

    @staticmethod
    def covid_non_covid(df):
        """
        INIT STEP: 
        Split data for COVID vs NON-COVID
        Return covid non covid dfs
        """

        # Convert df index to TimeStamp if the index is just seconds
        if df.index.dtype == "int64":
            df.index = pd.to_datetime(df.index, unit='s')
            
        covid_df = df.loc['2020-02-27 00:00:00' :, :]
        non_covid_df = df.loc[: '2020-02-26 23:59:45', :]

        # Reset index 
        covid_df.reset_index()
        non_covid_df.reset_index()
        
        return covid_df, non_covid_df

    @staticmethod
    def user_period_split(df, start_period, end_period):
        """
        Parse the period of the df according to the user's desire
        """
        # Convert df index to TimeStamp if the index is just seconds
        if df.index.dtype == "int64":
            df.index = pd.to_datetime(df.index, unit='s')
            
        user_df = df.loc[start_period : end_period, :]

        return user_df

    @staticmethod 
    def cpu_gpu(df):
        """
        SECOND STEP:
        Split the nodes, CPU vs GPU
        Return the cpu, and gpu partitioned dfs
        """

        cpu_nodes = [cpu_node for cpu_node in df.columns if cpu_node[0:5] in CPU_RACKS]
        gpu_nodes = [gpu_node for gpu_node in df.columns if gpu_node not in cpu_nodes]

        return df[cpu_nodes], df[gpu_nodes]

    @staticmethod
    def get_rack_nodes(df, my_rack):
        rack_nodes = set()

        for node in df.columns:
            rack = node.split("n")[0]
            if rack == my_rack:
                rack_nodes.add(node)

        return df.loc[: , rack_nodes]

    @staticmethod
    def nr_cores(self):
        """
        Break down number of cores for both CPU and GPU
        This function should be used inside the cpu_gpu. 
        Becuase, number of cores should be identified after splitting the nodes.
        Although, vice versa is also possible.
        """
        pass

    @staticmethod
    def type_of_procs(self):
        """
        Split from types of processors, there are 5 to 10 types.
        """
        pass