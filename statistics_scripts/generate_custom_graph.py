import matplotlib.pyplot as plt
import sys, os
from pathlib import Path
import matplotlib.pylab as pylab
import numpy as np
import pandas as pd

sys.path.insert(1, '/Users/cetinmehmet/Desktop/surfsara-tool/statistics_scripts')
sys.path.insert(2, '/Users/cetinmehmet/Desktop/surfsara-tool/parser')
sys.path.insert(3, '/Users/cetinmehmet/Desktop/surfsara-tool/analysis')


DAY = 24
MID_DAY = int(DAY / 2)
WEEK = 7 * DAY
TOOL_PATH = Path(os.path.abspath(__file__)).parent.parent
MARKERS = ['s', '*', 'o', 'v', '<', 'p', '.', 'd']
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    

# Configure label sizes of graphs
params = {
    'xtick.labelsize':16,
    'ytick.labelsize':16,
    'axes.labelsize':18,
    'figure.figsize': (10, 8),
    'savefig.format': 'pdf',
    'axes.titlesize': 20,
    'legend.loc': 'best',
    'legend.fontsize': "large"
}

pylab.rcParams.update(params)


class GenerateCustomGraph:

    def __init__(self, title, savefig_title, **kargs):
        self.title = title
        self.savefig_title = savefig_title
        self.ylabel = kargs['ylabel'] 
        self.period = kargs['period'] if kargs['period'] else print("No period specified")
        if self.period == "FULL":
            self.timestamp = " full season "
        elif self.period == None:
            self.timestamp = ""
        else:
            self.timestamp = str(" " + self.period[0].strftime("%Y-%m-%d") + " to " + self.period[1].strftime("%Y-%m-%d")) 


    ##### CUSTOM ANALYSIS ###################
    def entire_period_analysis(self, df_dict):
        def __get_converted_xticks(self, ax):
            """
            :param ax:
            :return list of day strings
            """
            return [pd.to_datetime(tick, unit='d').date().strftime("%d\n%b") for tick in ax.get_xticks()]
            
        def ax_components(ax):
            # Set other features of plot
            ax.set_ylim(0, )
            ax.set_xlabel("Days")
            ax.set_ylabel(self.ylabel)
            ax.set_title(self.title)
            ax.legend(loc="upper left")
            ax.set_xticklabels(labels=self.__get_converted_xticks(ax))

        _, (ax) = plt.subplots( figsize=(18,10))
        df_keys = []

        # Get the df keys passed
        for k in df_dict:
            if df_dict[k] != None:
                df_keys.append(k)


        # Nodes specified: Custom nodes; Custom period
        if 'df_custom' in df_keys:
            df = df_dict['df_custom'][0] # Must remove the brackets by getting [0] of list
            df.index = pd.to_datetime(df.index, unit='s')
            df.sort_index(inplace=True)

            fig, ax_arr = plt.subplots(len(df.columns), 1, sharex=True, constrained_layout=True, figsize=(30, 25))

            for i in range(len(df.columns)):                
                curr_node = df.iloc[:, i:i+1]
                ax_arr[i].plot(curr_node, label=curr_node.columns[0], color=COLORS[i])
                mean_val = round(curr_node.mean(axis=0).values[0], 2)
                median_val = round(curr_node.median(axis=0).values[0], 2)
                median_val2 = round(curr_node[curr_node.values > 0].median(axis=0).values[0], 2)
                ax_arr[i].axhline(y=mean_val, c='black', ls=':', lw=4, label="mean: " + str(mean_val))
                ax_arr[i].axhline(y=median_val, c='black', ls='--', lw=4, label="median: " + str(median_val))
                ax_arr[i].axhline(y=median_val2, c='gray', ls='-', lw=4, label="median (zeros filtered): " + str(median_val2))
                ax_components(ax_arr[i])

            self.title += str(" custom nodes")
               

         # Rack specified
        elif 'df_rack' in df_keys:
            _, ax = plt.subplots(1, 1)
            df = df_dict['df_rack'][0]
            df.sort_index(inplace=True)
            df.index = pd.to_datetime(df.index, unit='s')

            df_aggr = df.aggregate(func=sum, axis=1) # Aggregate the nodes in the rack
            ax.plot(df_aggr, color=COLORS[1], label=str(df.columns[0].split("n")[0]) + " aggregated load1")
            ax_components(ax)

        # Custom period; nodes are default CPU vs GPU
        elif 'df_cpu' in df_keys:
            df_cpu = df_dict['df_cpu'][0]
            df_gpu = df_dict['df_gpu'][0]

            df_cpu.index = pd.to_datetime(df_cpu.index, unit='s')
            df_gpu.index = pd.to_datetime(df_gpu.index, unit='s')

            # Pass the mean of the nodes
            df_cpu_mean = df_cpu.mean(axis=1)
            df_gpu_mean = df_gpu.mean(axis=1)

            ax.plot(df_cpu_mean, label="CPU", color=COLORS[0])
            ax.plot(df_gpu_mean, label="GPU", color=COLORS[1])
            ax_components(ax)

        # Period not specified
        elif 'df_covid' in df_keys:
            print("Not possible for this analysis type")

        # Nodes not specified
        elif 'df_cpu_covid' in df_keys:
            print("Not possible for this analysis type")
       
        self.savefig_title += "entire_period"
        plt.savefig(os.path.join(str(TOOL_PATH) + "/plots/" + self.savefig_title + ".pdf"), dpi=100) 
        plt.show()
        plt.pause(0.0001)

    def custom_daily_seasonal_diurnal_pattern(self, df_dict):
        def get_time_df(df):
            df["dt"] = pd.to_datetime(df.index, utc=True, unit="s")
            df["hour"] = df["dt"].dt.hour
            df["day"] = df["dt"].apply(lambda x: x.weekday())

            df = df.groupby(["day", "hour"]).mean()
            df.index = [hour for hour in range(0, 24*7)]
            return df

        def ax_components(ax):
            ax.set_title(self.title)
            ax.set_ylabel(self.ylabel)
            ax.set_ylim(0, )
            ax.set_xlabel("Days")
            ax.set_xticks([tick for tick in range(MID_DAY-1, WEEK, DAY)])
            ax.set_xticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
            ax.legend(loc="upper right")
            xcoords = [0] + [xcoord for xcoord in range(23, WEEK, DAY)]
            for xc in xcoords:
                ax.axvline(x=xc, color="gray", lw=0.5)

        df_keys = []

        # Get the df keys passed
        for k in df_dict:
            if df_dict[k] != None:
                df_keys.append(k)

        # Nodes specified: Custom nodes; Custom period
        if 'df_custom' in df_keys:
            df = df_dict['df_custom'][0]
            df.sort_index(inplace=True)
            df = get_time_df(df)
            
            fig, ax_arr = plt.subplots(len(df.columns), 1, sharex=True, constrained_layout=True)

            for i in range(len(df.columns)):
                # Must remove the brackets by getting [0] of list
                curr_node = df.iloc[:, i:i+1]
                ax_arr[i].plot(curr_node, label=curr_node.columns[0], color=COLORS[i], marker=MARKERS[i])
                ax_components(ax_arr[i])

            self.title += str(" custom nodes")

        # Nodes covid non covid
        if 'df_covid' in df_keys:
            df_covid = df_dict['df_covid'][0]
            df_non_covid = df_dict['df_non_covid'][0]

            df_covid = get_time_df(df_covid)
            df_non_covid = get_time_df(df_non_covid)

            fig, ax_arr = plt.subplots(len(df_covid.columns), 1, sharex=True, constrained_layout=True)

            for i in range(len(df_covid.columns)):
                ax_arr[i].plot(df_covid.iloc[:, i:i+1], label=df_covid.iloc[:, i:i+1].columns[0] + " covid", color=COLORS[0], marker=MARKERS[0])
                ax_arr[i].plot(df_non_covid.iloc[:, i:i+1], label=df_non_covid.iloc[:, i:i+1].columns[0] + " non-covid", color=COLORS[1], marker=MARKERS[1])
                ax_components(ax_arr[i])

        # Rack specified
        elif 'df_rack' in df_keys:
            df = df_dict['df_rack'][0]
            df.sort_index(inplace=True)        
            df = get_time_df(df)
            df_aggr = df.aggregate(func=sum, axis=1) # Aggregate the nodes in the rack

            _, ax = plt.subplots()
            ax.plot(df_aggr, label=df.columns[0].split("n")[0], color=COLORS[0], marker=MARKERS[0])
            ax_components(ax)
            
        elif 'df_rack_covid' in df_keys:
            df_covid = df_dict['df_rack_covid'][0]
            df_non_covid = df_dict['df_rack_non_covid'][0]

            df_covid = get_time_df(df_covid)
            df_non_covid = get_time_df(df_non_covid)

            df_covid_aggr = df_covid.aggregate(func=sum, axis=1) # Aggregate the nodes in the rack 
            df_non_covid_aggr = df_non_covid.aggregate(func=sum, axis=1) # Aggregate the nodes in the rack 

            _, ax = plt.subplots()
            ax.plot(df_covid_aggr, label=df_covid.columns[0].split("n")[0] + " covid", c=COLORS[0], marker=MARKERS[0])
            ax.plot(df_non_covid_aggr, label=df_non_covid.columns[0].split("n")[0] + " non-covid", c=COLORS[1], marker=MARKERS[1])
            ax_components(ax)

        # Custom period; nodes are default CPU vs GPU
        elif 'df_cpu' in df_keys:
            df_cpu = df_dict['df_cpu'][0]
            df_gpu = df_dict['df_gpu'][0]

            df_cpu = get_time_df(df_cpu)
            df_gpu = get_time_df(df_gpu)

            df_cpu_aggr = df_cpu.aggregate(func=sum, axis=1)
            df_gpu_aggr = df_gpu.aggregate(func=sum, axis=1)

            _, ax = plt.subplots()  
            ax.plot(df_cpu_aggr, label="CPU", color=COLORS[2], marker=MARKERS[2])
            ax.plot(df_gpu_aggr, label="GPU", color=COLORS[3], marker=MARKERS[3])
            ax_components(ax)

            self.title += " CPU vs GPU nodes"

        self.title += self.timestamp
 
        self.savefig_title += "custom_daily_seasonal"
        plt.savefig(os.path.join(str(TOOL_PATH) + "/plots/" + self.savefig_title + ".pdf"), dpi=100) 
        plt.show()
        plt.pause(0.0001)


    def custom_hourly_seasonal_diurnal_pattern(self, df_dict):

        def get_time_df(df):
            df["dt"] = pd.to_datetime(df.index, utc=True, unit="s")
            df["hour"] = df["dt"].dt.hour

            df = df.groupby("hour").mean()
            return df

        def ax_components(ax):
            ax.set_xticks([i for i in range(24)], minor=True)
            ax.tick_params('x', length=12, width=2, which='major')
            ax.tick_params('x', length=8, width=1, which='minor')
            ax.set_title(self.title)
            ax.set_ylabel(self.ylabel)
            ax.set_ylim(0, )
            ax.set_xlabel("Hours")
            ax.legend(loc="upper right")


        df_keys = []

        # Get the df keys passed
        for k in df_dict:
            if df_dict[k] != None:
                df_keys.append(k)

        # Nodes specified: Custom nodes; Custom period
        if 'df_custom' in df_keys:
            df = df_dict['df_custom'][0]
            df.sort_index(inplace=True)

            df = get_time_df(df)
            
            fig, ax_arr = plt.subplots(len(df.columns), 1, sharex=True, constrained_layout=True)
            for i in range(len(df.columns)):
                # Must remove the brackets by getting [0] of list
                curr_node = df.iloc[:, i:i+1]
                ax_arr[i].plot(curr_node, label=curr_node.columns[0], color=COLORS[i], marker=MARKERS[i])
                ax_components(ax_arr[i])

            self.title += str(" custom nodes")

                # Nodes covid non covid
        elif 'df_covid' in df_keys:
            df_covid = df_dict['df_covid'][0]
            df_non_covid = df_dict['df_non_covid'][0]

            df_covid = get_time_df(df_covid)
            df_non_covid = get_time_df(df_non_covid)
            
            fig, ax_arr = plt.subplots(len(df_covid.columns), 1, sharex=True, constrained_layout=True)
            for i in range(len(df_covid.columns)):
                ax_arr[i].plot(df_covid.iloc[:, i:i+1], label=df_covid.iloc[:, i:i+1].columns[0] + " covid", color=COLORS[0], marker=MARKERS[0])
                ax_arr[i].plot(df_non_covid.iloc[:, i:i+1], label=df_non_covid.iloc[:, i:i+1].columns[0] + " non-covid", color=COLORS[1], marker=MARKERS[1])
                ax_components(ax_arr[i])

         # Rack specified
        elif 'df_rack' in df_keys:
            df = df_dict['df_rack'][0]
            df.sort_index(inplace=True)
            
            df = get_time_df(df)
            df_aggr = df.aggregate(func=sum, axis=1) # Aggregate the nodes in the rack
            
            _, ax = plt.subplots()
            ax.plot(df_aggr, label=df.columns[0].split("n")[0], color=COLORS[0], marker=MARKERS[0])
            ax_components(ax)

        elif 'df_rack_covid' in df_keys:
            df_covid = df_dict['df_rack_covid'][0]
            df_non_covid = df_dict['df_rack'][0]

            df_covid = get_time_df(df_covid)
            df_non_covid = get_time_df(df_non_covid)

            df_covid_aggr = df_covid.aggregate(func=sum, axis=1) # Aggregate the nodes in the rack 
            df_non_covid_aggr = df_non_covid.aggregate(func=sum, axis=1) # Aggregate the nodes in the rack 

            _, ax = plt.subplots()
            ax.plot(df_covid_aggr, label=df.columns[0].split("n")[0] + " covid", c=COLORS[0], marker=MARKERS[0])
            ax.plot(df_non_covid_aggr, label=df.columns[0].split("n")[0] + " non-covid", c=COLORS[1], marker=MARKERS[1])
            ax_components(ax)

        # Custom period; nodes are default CPU vs GPU
        elif 'df_cpu' in df_keys:
            df_cpu = df_dict['df_cpu'][0]
            df_gpu = df_dict['df_gpu'][0]

            df_cpu = get_time_df(df_cpu)
            df_gpu = get_time_df(df_gpu)

            df_cpu = df_cpu.aggregate(func=sum, axis=1)
            df_gpu = df_gpu.aggregate(func=sum, axis=1)

            _, ax = plt.subplots()
            ax.plot(df_cpu, label="CPU", color=COLORS[2], marker=MARKERS[2])
            ax.plot(df_gpu, label="GPU", color=COLORS[3], marker=MARKERS[3])
            ax_components(ax)

            self.title += " CPU vs GPU nodes aggregated values "

        self.title += self.timestamp

        self.savefig_title += "custom_hourly_seasonal"
        plt.savefig(os.path.join(str(TOOL_PATH) + "/plots" + self.savefig_title + ".pdf"), dpi=100) 
        plt.show()
        plt.pause(0.0001)

    def custom_cdf(self, df_dict):
        df_keys = []

        # Get all the values in the df
        def get_custom_values(df):
            values = np.array([])
            for column in df.columns:
                arr = df[column].values
                mask = (np.isnan(arr) | (arr < 0))
                arr = arr[~mask]  # Filter out NaN values and less than 0
                values = np.append(values, arr)

            return values

        def ax_components(ax):
            ax.set_title(self.title)
            ax.set_ylabel("Density")
            ax.set_xlim(0, )
            ax.set_xlabel(self.ylabel)
            ax.legend(loc='lower right')

        self.title = "CDF " + self.title

        # Get the df keys passed
        for k in df_dict:
            if df_dict[k] != None:
                df_keys.append(k)

        # Nodes specified: Custom nodes; Custom period
        if 'df_custom' in df_keys:
            df = df_dict['df_custom'][0]
            df.sort_index(inplace=True)
            self.title += self.timestamp
            fig, ax_arr = plt.subplots(len(df.columns), 1, sharex=True, constrained_layout=True)
            for i in range(len(df.columns)):
                # Must remove the brackets by getting [0] of list
                curr_node = df.iloc[:, i:i+1]
                ax_arr[i].hist(get_custom_values(curr_node), label=curr_node.columns[0], color=COLORS[i], density=True, histtype='step', bins=100, cumulative=True)
                ax_components(ax_arr[i])

        # Nodes covid non covid
        elif 'df_covid' in df_keys:
            df_covid = df_dict['df_covid'][0]
            df_non_covid = df_dict['df_non_covid'][0]
            
            fig, ax_arr = plt.subplots(len(df_covid.columns), 1, sharex=True, constrained_layout=True)
            for i in range(len(df_covid.columns)):
                ax_arr[i].hist(get_custom_values(df_covid.iloc[:, i:i+1]), label=df_covid.iloc[:, i:i+1].columns[0] + " covid", color=COLORS[0], density=True, histtype='step', bins=100, cumulative=True)
                ax_arr[i].hist(get_custom_values(df_non_covid.iloc[:, i:i+1]), label=df_non_covid.iloc[:, i:i+1].columns[0] + " non-covid", color=COLORS[1], density=True, histtype='step', bins=100, cumulative=True)
                ax_components(ax_arr[i])

         # Rack specified
        elif 'df_rack' in df_keys:
            self.title += self.timestamp
            df = df_dict['df_rack'][0]
            df.sort_index(inplace=True)
            df_values = get_custom_values(df) # get all the values in df
            
            _, ax = plt.subplots()
            ax.hist(df_values, label=df.columns[0].split("n")[0], color=COLORS[0], density=True, histtype='step', bins=100, cumulative=True)
            ax_components(ax)

        elif 'df_rack_covid' in df_keys:
            df_covid = df_dict['df_rack_covid'][0]
            df_non_covid = df_dict['df_rack'][0]

            df_covid_values = get_custom_values(df_covid) # Aggregate the nodes in the rack 
            df_non_covid_values = get_custom_values(df_non_covid) # Aggregate the nodes in the rack 

            _, ax = plt.subplots()
            ax.hist(df_covid_values, label=df.columns[0].split("n")[0] + " covid", c=COLORS[0], density=True, histtype='step', bins=100, cumulative=True)
            ax.hist(df_non_covid_values, label=df.columns[0].split("n")[0] + " non-covid", c=COLORS[1], density=True, histtype='step', bins=100, cumulative=True)
            ax_components(ax)

        # Custom period; nodes are default CPU vs GPU
        elif 'df_cpu' in df_keys:
            self.title += " CPU vs GPU nodes all values " + self.timestamp
            df_cpu = df_dict['df_cpu'][0]
            df_gpu = df_dict['df_gpu'][0]

            df_cpu = get_custom_values(df_cpu)
            df_gpu = get_custom_values(df_gpu)

            _, ax = plt.subplots()
            ax.hist(df_cpu, label="CPU", color=COLORS[2], density=True, histtype='step', bins=1000, cumulative=True)
            ax.hist(df_gpu, label="GPU", color=COLORS[3], density=True, histtype='step', bins=1000, cumulative=True)
            ax_components(ax)

        self.title += self.timestamp
        self.savefig_title += "cdf"
        plt.savefig(os.path.join(str(TOOL_PATH) + "/plots/" + self.savefig_title + ".pdf"), dpi=100) 
        plt.show()