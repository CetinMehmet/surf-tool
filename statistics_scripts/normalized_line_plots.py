import pandas as pd


class NormalizedPlot:

    @staticmethod
    def get_plot(df: pd.DataFrame, title: str):
        df_mean = df.mean(axis=1)
        norm_df = df_mean / max(df_mean) # Normalize according to the highest value in the df
        norm_df["dt"] = pd.to_datetime(norm_df.index, unit='s')
        norm_df.groupby("dt").mean().mean(axis=1)

        # Plot line graph
        ax = norm_df.plot.line(
            x=norm_df.index,
            y=norm_df.values,
            rot=45,
            color="blue"
        )
        ax.set_ylim(0, 1.05) # Set limit to 1.05 for the margin to be not too tight
        ax.set_title(title)


