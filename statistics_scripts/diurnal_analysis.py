import pandas as pd
import seaborn as sns
import pytz

#TODO: This class needs to be changed a lot. The average of average is wrong.

class DiurnalAnalysis:

    def __init__(self, df, df5=None, df15=None):
        self.df = df
        self.df5 = df5
        self.df15 = df15

    # Written by: Laurens Versluis
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

    def __hourly_analysis(self, df, title: str):
        df_per_hour_per_node = self.df.groupby("hour_of_day").mean()
        df_per_hour = df_per_hour_per_node.mean(axis=1)

        sns_plot = sns.barplot(x=df_per_hour.index, y=df_per_hour.values)
        sns_plot.set_xlabel("Hour of day")
        sns_plot.set_ylabel("Entropy available bytes")
        sns_plot.set_title("Hourly Entropy available bytes")

    def __daily_analysis(self, df, title: str):
        df_per_hour_per_node = self.df.groupby("day").mean()
        df_per_hour = df_per_hour_per_node.mean(axis=1)

        sns_plot = sns.barplot(x=df_per_hour.index, y=df_per_hour.values)
        sns_plot.set_xlabel("Hour of day")
        sns_plot.set_ylabel("Entropy available bytes")
        sns_plot.set_title("Hourly Entropy available bytes")

    def __monthly_analysis(self, df, title: str):
        df_per_hour_per_node = self.df.groupby("month").mean()
        df_per_hour = df_per_hour_per_node.mean(axis=1)

        sns_plot = sns.barplot(x=df_per_hour.index, y=df_per_hour.values)
        sns_plot.set_xlabel("Hour of day")
        sns_plot.set_ylabel("Entropy available bytes")
        sns_plot.set_title("Hourly Entropy available bytes")

    def __full_time_analysis(self, df, title: str):
        df_per_date_per_node = self.df.groupby("date").mean()
        df_per_date = df_per_date_per_node.mean(axis=1)

        # Plot line graph
        ax = df_per_date.plot.line(
            x=df.index,
            y=df.values,
            rot=45,
            color="blue"
        )
        ax.set_ylim(0, )
        ax.set_title(title)

    def plot_analysis(self, time_frame: str, title: str):
        self.df = self.get_diurnal_df(self.df)

        if time_frame == "hour":
            self.__hourly_analysis(self.df, title)
        elif time_frame == "day":
            self.__daily_analysis(self.df, title)
        elif time_frame == "month":
            self.__monthly_analysis(self.df, title)
        else:
            self.__full_time_analysis(self.df, title)






