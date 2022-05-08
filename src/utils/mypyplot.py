#!/usr/bin/python3.5

"""
    The ``singalun.mypyplot`` module
    ================================
    This module contains basic plotting methods that are often reused in 
    different singalun modules / libraries. They mostly rely on matplotlib.

    Details
    -------
    * Author: Leon-Charles Tranchevent.
    * Version: 0.1
    * Date: 2017/01 
"""

# ============================================================================
#
#       CONFIGURATION
#
# ============================================================================

# External libraries.
import matplotlib.pyplot as plt
import numpy as np

# Configure libraries.
plt.style.use('seaborn-colorblind')

# ============================================================================
#
#       FUNCTIONS - UTILS
#
# ============================================================================

def compute_axis_lim(plot_data, global_min = 0, global_max = 1, 
    border = .05):

    """
        Compute the optimal axis values.

        Optimal axis values are often required to efficiently plot an array. 
        This function computes them by taking the minimum and maximum over a
        provided array that contains the values to be plotted. The minimum and 
        maximum values are corrected with the global minimum and maximum values 
        if necessary. In addition, a small border can be added to better render 
        the plot.
 
        :param plot_data: The array that will be plotted.
        :param global_min: The global axis minimum that can not be exceeded 
            (default is 0)
        :param global_max: The global axis maximum that can not be exceeded 
            (default is 1).
        :param border: The border to add on each side for a better rendering 
            (default is .05).
        :type plot_data: numpy.ndarray(float)
        :type global_min: float
        :type global_max: float
        :type border: float
        :return: The computed axis limits (minimum and maximum).
        :rtype: list(float, float)
 
        :Example:
 
        >>> compute_axis_lim(numpy.array([0, 1]))
        (-0.05, 1.05)
        >>> compute_axis_lim(numpy.array([0, 2]), global_min = 1, 
            global_max = 2, border = 0)
        (1, 2)

        .. note:: This function has mostly been designed for internal purposes.
    """

    # We compute the minimum and maximum of the arrays but replace by the 
    # global maximum and minimum when necessary, borders are also added.
    axis_lim_minimum = max(global_min, plot_data.min()) - border
    axis_lim_maximum = min(global_max, plot_data.max()) + border

    # We return the values
    return axis_lim_minimum, axis_lim_maximum

# ============================================================================
#
#       FUNCTIONS - PLOT CONFIGURATION
#
# ============================================================================

def setaxis():

    """
        Refine the axis of an existing figure.

        This function changes the current x- and y-axis to remove the ticks
        that are on top and on the right (for convenience), but keeps the box 
        around the whole plot.
 
        .. note:: This function has mostly been designed for internal purposes.
    """

    # We set the axis to have only ticks on the left and bottom axis.
    plt.tick_params(axis = 'x', which = 'both', bottom = 'on', top = 'off', 
        labelbottom = 'on')
    plt.tick_params(axis = 'y', which = 'both', left = 'on', right = 'off', 
        labelleft = 'on')

# ============================================================================
#
#       FUNCTIONS - HISTOGRAMS
#
# ============================================================================

def plot_hist(figure_gridspec, subplot_number, plot_data, plot_title = '', 
    background_color = 'white', keep_grid = False, nb_bins = 0):

    """
        Plot the value distribution from an array.

        This function can be used to plot an histogram of values that are 
        stored in a numpy array. The function assumes that the figure already 
        exists, and that its gridspec has been defined. The number of bins of 
        the is histogram configurable.
 
        :param figure_gridspec: The gridspec object linked to the existing 
            figure that should be used for the current subplot.
        :param subplot_number: The subplot number to use for the current 
            subplot.
        :param plot_data: The array whose value distribution is to plot.
        :param plot_title: The title of the plot (default is '').
        :param background_color: The background color of the figure (default 
            is white).
        :param keep_grid: A boolean that indicates whether to keep the grid 
            (default is False).
        :param nb_bins: The number of bins for the histogram (default is 0, 
            which let the program decides on the optimal number of bins).
        :type figure_gridspec: matplotlib.gridspec
        :type subplot_number: int
        :type plot_data: numpy.ndarray(float)
        :type plot_title: str
        :type background_color: str
        :type keep_grid: bool
        :type nb_bins: int 
    """

    # We prepare the subplot.
    plt.subplot(figure_gridspec[subplot_number], axisbg = background_color)

    # We plot the histogram itself.
    if nb_bins > 0:
        plt.hist(plot_data, bins = nb_bins)
    else:
        plt.hist(plot_data)

    # We configure the subplot (axis, grid, title).
    setaxis()
    plt.grid(keep_grid)
    plt.title(plot_title)

# ============================================================================
#
#       FUNCTIONS - SCATTERS
#
# ============================================================================

def plot_scatter_flatarrays(figure_gridspec, subplot_number, first_plot_data, 
    first_plot_data_tag, second_plot_data, second_plot_data_tag, 
    background_color = 'white', keep_grid = False, plot_title = '', 
    xmin = -1, xmax = 1, ymin = -1, ymax = 1, 
    alpha = 0.05):

    """
        Build a scatter plot using two arrays.

        This function can be used to build a scatter plot of a variable 
        stored in a first array) against another variable (stored in a second 
        array). This functions assumes that the figure already exists, and that 
        its grid spec has been defined.

        :param figure_gridspec: The gridspec object linked to the existing 
            figure that should be used for the current subplot.
        :param subplot_number: The subplot number to use for the current 
            subplot.
        :param first_plot_data: The first array to use for the scatter plot.
        :param first_plot_data_tag: The tag associated with the first array
            (used for the axis label).
        :param second_plot_data: The second array to use for the scatter plot.
        :param second_plot_data_tag: The tag associated with the second array
            (used for the axis label).
        :param background_color: The background color of the figure (default 
            is white).
        :param keep_grid: A boolean that indicates whether to keep the grid 
            (default is False).
        :param plot_title: The title of the plot (default is '').
        :param xmin: The minimum value of the x-axis (default is -1).
        :param xmax: The maximum value of the x-axis (default is 1).
        :param ymin: The minimum value of the y-axis (default is -1).
        :param ymax: The maximum value of the y-axis (default is 1).
        :param alpha: The alpha transparency value (default is 0.05).
        :type figure_gridspec: matplotlib.gridspec
        :type subplot_number: int
        :type first_plot_data: numpy.ndarray(float)
        :type first_plot_data_tag: str
        :type second_plot_data: numpy.ndarray(float)
        :type second_plot_data_tag: str
        :type background_color: str
        :type keep_grid: bool
        :type plot_title: str
        :type xmin: float
        :type xmax: float
        :type ymin: float
        :type ymax: float
        :type alpha: float
    """

    # We prepare the subplot.
    plt.subplot(figure_gridspec[subplot_number], axisbg = background_color)

    # We remove the grid if necessary.
    plt.grid(keep_grid)

    # We plot the diagonal as a guide to the eye.
    plt.plot([xmin, xmax], [ymin, ymax], 'k-.')

    # We plot the data as a scatter plot.
    plt.scatter(first_plot_data, second_plot_data, alpha = alpha)

    # We set the axis limits.
    x_minimum, x_maximum = compute_axis_lim(first_plot_data, xmin, 
        xmax, .05)
    y_minimum, y_maximum = compute_axis_lim(second_plot_data, ymin, 
        ymax, .05)
    plt.xlim([x_minimum, x_maximum])
    plt.ylim([y_minimum, y_maximum])

    # We compute the the best fit.
    polyfit_z = np.polyfit(first_plot_data, second_plot_data, 1)
    polyfit_p = np.poly1d(polyfit_z)
    r_squared = 1 - (sum((second_plot_data - (polyfit_z[0] * first_plot_data + 
        polyfit_z[1])) ** 2) / ((len(second_plot_data) - 1) 
        * np.var(second_plot_data, ddof = 1)))

    # We plot the best fit.
    plt.text(x_minimum + (x_maximum - x_minimum) * .15, 
        y_minimum + (y_maximum - y_minimum) * 0.85, 
        'R2 = ' + str(round(r_squared, 2)), fontsize = 14)
    plt.plot(first_plot_data, polyfit_p(first_plot_data), '-')

    # We set the axis labels and the title.
    plt.xlabel(first_plot_data_tag)
    plt.ylabel(second_plot_data_tag)
    plt.title(plot_title + second_plot_data_tag + ' vs ' + first_plot_data_tag)
