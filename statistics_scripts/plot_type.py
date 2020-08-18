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
