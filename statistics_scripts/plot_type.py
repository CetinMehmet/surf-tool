from scipy.stats.stats import pearsonr, spearmanr, kendalltau

"""
    Since color blinds may also be using the tool, no red or green colors shall be used for any plot
"""

class PlotType:

    @staticmethod
    def cdf(ax, dataset, color, label=None, xlabel=None, ylabel=None, title=None):
        ax.hist(dataset, bins=100, density=True, cumulative=True, histtype="step", color=color, label=label)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    @staticmethod
    def hist(ax, dataset, color=None, label=None, title=None, ylabel=None, xlabel=None):
        ax.hist(dataset, bins=100, color=color, label=label)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    @staticmethod
    def normalized_lineplot():
        pass

    """
        Spearman:
            - Is there a statistically significant relationship between participants
            - The assumptions of the Spearman correlation are that data must be at least ordinal,
            and the scores on one variable must be monotonically related to the other variable.

        Pearson:
            - Is used to measure the degree of the relationship between linearly related variables.
            - Is there a statistically significant relationship.
            - For the Pearson r correlation, both variables should be normally distributed,
            (normally distributed variables have a bell-shaped curve).
            Other assumptions include linearity and homoscedasticity.
            Linearity assumes a straight line relationship between each of the two variables,
            and homoscedasticity assumes that data is equally distributed about the regression line.

        Kendall:
            - is a non-parametric test that measures the strength of dependence between two variables.
    """
    @staticmethod
    def scatter_plot():
        pass
