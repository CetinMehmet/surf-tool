
"""
    Split the metrics in terms of:
        1. Plot the entire set of nodes, which means no partition in nodes
        2. Split the nodes, CPU vs GPU
        3. Break down number of cores for both CPU and GPU
        4. Split from types of processors, there are 5 to 10 types.
        5. COVID vs NON-COVID
"""

class ParseMetric:

    def __init__(self):
        pass

    def covid_non_covid(self):
        """
        INIT STEP: 
        Split data for COVID vs NON-COVID
        """
        pass

    def cpu_gpu(self):
        """
        SECOND STEP:
        Split the nodes, CPU vs GPU
        """
        pass

    def nr_cores(self):
        """
        Break down number of cores for both CPU and GPU
        This function should be used inside the cpu_gpu. 
        Becuase, number of cores should be identified after splitting the nodes.
        Although, vice versa is also possible.
        """
        pass

    def type_of_procs(self):
        """
        Split from types of processors, there are 5 to 10 types.
        """
        pass