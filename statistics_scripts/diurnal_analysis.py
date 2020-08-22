import pandas as pd
import seaborn as sns
import pytz

"""
    Diurnal Analysis:
        1. Hourly analysis:
            a. "Aggregated" over the entire period
            b. "Aggregated" per month 

            The hourly analysis allows us to see differences in office hours (9-5) vs non-office hours; 
            Did the covid period affect people’s working habits (did they do more outside office hours for example?)

        2. Daily analysis:
            a. "Aggregated" over the entire period
            b. "Aggregate" per month

        3. Workday vs weekend:
            a. "Aggregated" over the entire period
            b. "Aggregate" per month
        
        4. Monthly (seasonal patterns):
            a. aggregate all data per month (for each metric → one value per month, 
            or the basic statistics (mean, min, median, max, other quartiles, stddev, CoV if possible))

        5. Per node per metric create a plot to inspect per node.

        6. Inspect Eigen values per node/metric.

"""

class DiurnalAnalysis:

    def __init__(self, df, df5=None, df15=None):
        self.df = df
        self.df5 = df5
        self.df15 = df15

    # Written by `Laurens Versluis`
    @staticmethod
    def get_diurnal_df(df):
        df = df.loc[:, (df.max() > 0)]

        # Parse all times to UTC datetime objects
        # VERY IMPORTANT! By default pandas assumes nanoseconds as units, very annoying to debug
        # if you do not set unit="s" as you will get duplicate index issues...
        df["dt"] = pd.to_datetime(df.index, utc=True, unit="s")
        # Convert everything into localized Amsterdam time and then drop the timezone info again
        # dropping it is required to save the parquet file.
        df["dt"] = df["dt"].dt.tz_convert(pytz.timezone('Europe/Amsterdam')).dt.tz_localize(None)
        # Get hour of day and day columns to plot :)
        df["hour_of_day"] = df["dt"].dt.hour
        df["day"] = df["dt"].apply(lambda x: x.weekday())

        return df

    def hourly(self, df, title: str):
        """
        Hourly analysis:
            a. "Aggregated" over the entire period
            b. "Aggregated" per month 
        """
        pass

    def daily(self, df, title: str):
        """
        Daily analysis:
            a. "Aggregated" over the entire period
            b. "Aggregate" per month
        """
        pass

    def monthly(self, df, title: str):
        """
        Monthly (seasonal patterns):
            a. "Aggregate" all data per month (for each metric → one value per month, 
                or 
            b. the basic statistics (mean, min, median, max, other quartiles, stddev, CoV if possible))
        """
        pass

    def workday_weekend(self):
        """
        Workday vs weekend:
            a. "Aggregated" over the entire period
            b. "Aggregate" per month
        """
        pass

    def entire_period(self, df, title: str):
        """
        Per node per metric create a plot to inspect per node.
        """

    def plot_analysis(self, time_frame: str, title: str):
        """
        Plot requested analysis
        """
        pass






