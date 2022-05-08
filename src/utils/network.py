#!/usr/bin/python3.5

"""
    The ``singalun.network`` module
    ==================================
    This module contains the methods to go from a clean data matrix to a
    network based on a correlation analysis. It also contains the methods to
    perform network-based analyses.

    Details
    -------
    * Authors: Leon-Charles Tranchevent, Francisco Azuaje.
    * Version: 0.1
    * Date: 2017/01
"""

# ============================================================================
#
#       CONFIGURATION
#
# ============================================================================

# External libraries
from math import log
import networkx as nx
from scipy import stats
import pandas as p
import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt

# Own libraries
import mypyplot as mpp

# Configure libraries.
plt.style.use('seaborn-colorblind')

# Configure current libraries.
NB_NODES_APPROXIMATE_CFBC = 5000

# ============================================================================
#
#       FUNCTIONS - ADJACENCY MATRIX
#
# ============================================================================

def beta_gridsearch(data_corr, data_uptri, truncated = True, 
    skip_first_value = False, beta_grid = [2, 13, 2], r2_threshold = 0.9, 
    verbose = False):

#TODO write docstrings.
# TODO: write docstrings.
    # We assume that the correlation coefficients have been rescaled. 

    # We store the default values to be returned.
    beta_opt = 1
    data_adj_opt = data_corr
    r2_opt = -1

    # We can perform the second step of WGCNA, which consists in 
    # using the rescaled correlation and compute similarity measures using 
    # a power function with parameter beta.
    for beta in np.arange(beta_grid[0], beta_grid[1], beta_grid[2]):

        # We compute the similarities using beta, and the WGCNA formula to
        # produce an adjacency matrix. The regular formula is 
        #       data_adj = (.5 + .5 * data_corr) ** beta 
        # for values in [-1 ; 1], but since the data have already been 
        # standardized to [0;1] we use 
        #       data_adj = data_corr_std ** beta.
        data_adj = data_corr ** beta

        # We indicate that the adjacency matrix has been computed.
        if verbose is True:
            print('Adjacency matrix computed (Beta = ' + str(beta) + ').')

        # For a correct adjacency matrix, we set the diagonal to zeros in all
        # cases (although it does not matter in this case).
        np.fill_diagonal(data_adj.values, 0)

        # We select only the upper triangle of the matrix to check the 
        # scale-freeness of the associated network.
        # DataFrame.as_matrix is deprecated, should use DataFrame.values instead.
        data_adj_uptri = data_adj.values[data_uptri]

        # We discretize these values using a histogram.
        weighted_degree_discretized = plt.hist(data_adj_uptri.flatten(), 
            bins = 10)

        # The histogram basically contains the connectivity k and the 
        # associated probabilities p(k).
        # We compute log10(p(k))
        log_p_k = np.log10(weighted_degree_discretized[0][0:])

        # We can skip the first value that sometimes bias the estimation.
        if skip_first_value:
            log_p_k = log_p_k[1:]

        # We extract k and compute log10(k). Using hist, k is always one item 
        # larger than p(k), so we start at 1 instead of 0.
        k = weighted_degree_discretized[1][1:]

        # We can skip the first value that sometimes bias the estimation.
        if skip_first_value:
            k = k[1:]

        # We log it in any case.
        log_k = np.log10(k)

        # We prepare the regression.
        reg_a = np.vstack([np.ones(len(k)), log_k, k]).T

        # If we are in the truncated mode we try to fit 
        #       log(p(k)) ~ log(k) + k
        # otherwise, we try to fit the more regular power law
        #       log(p(k)) ~ log(k)
        if truncated is False:
            reg_a = np.vstack([np.ones(len(k)), log_k]).T

        # Filter out the infinite values.
        filter_inf = ~np.isinf(log_p_k).astype(bool)
        log_p_k = log_p_k[filter_inf]
        reg_a = reg_a[filter_inf]

        # We perform the regression.
        model, resid = np.linalg.lstsq(reg_a, log_p_k, rcond=-1)[:2]

        # We compute the r-squared.
        # Special case for strange data that does not give enough
        # points to be fit.
        if (len(resid) == 0):
            continue
        else:
            r2 = (1 - resid / (log_p_k.size * log_p_k.var()))[0]

        # We indicate that the scale free index has been computed.
        if verbose is True:
            print('Truncated scale free topology index computed. Model is ' 
                + str(model) + ' and r2 is ' + str(r2) + '.')

        # If we are above the threshold for r2, we consider we have found the 
        # optimal beta.
        beta_opt = beta
        data_adj_opt = data_adj
        r2_opt = r2
        if r2 >= r2_threshold:
            break

    # We issue a warning if we get out of the loop and r2 is still below the 
    # threshold.
    if r2_opt < r2_threshold:
        print('[Warning] The r2 value (' + str(r2_opt) + 
            ') has always been below the threshold (' + str(r2_threshold) + 
            ') for the given grid (' + str(beta_grid) + 
            '). You might consider extending the grid.')

    # We return the computed adjacency matrix, the optimal beta and the 
    # associated r2.
    return data_adj_opt, beta_opt, r2_opt

def compute_adjacency_matrix(data_asdf, method = 'pearson', 
    perform_wgcna = True, estimate_beta = True, beta_value = 12, 
    truncated = True, skip_first_value = False, beta_grid = [2, 13, 2], 
    r2_threshold = 0.9, verbose = False):

    """
        Compute the adjacency matrix of a network from a data matrix.

        This function takes a clean data matrix, computes the correlation 
        between the entities (columns), and rescale them to obtain a squared
        adjacency matrix. The values on the diagonal are set to 0 to avoid 
        self-loops.
 
        :param data_asdf: The clean dataframne to use.
        :param method: The correlation method to use among 'pearson', 
            'spearman', and 'kendall'.
        :param perform_wgcna: A boolean indicating whether WGCNA should be 
            performed, (default is True). This rescales the correlation values 
            [-1;1] to similarities [0;1] using the following formula:
            :math:`s = ((c - min(c)) / (max(c)-min(c)) ) ^ {B}`
            with the similarity :math:`s`, the correlation :math:`c`, and 
            the single parameter to estimate :math:`B` (Beta in WGCNA).
        :param estimate_beta: A boolean that states whether Beta should be 
            estimated automatically (default is True). The estimation is based
            on the scale-freeness of the network associated with the adjacency 
            matrix. 
        :param beta_value: The value for Beta, only used if perform_wgcna is 
            set to True and estimate_beta is set to False (default is 12).
        :param truncated: 
        :param skip_first_value: 
        :param beta_grid: 
        :param r2_threshold: 
        :param verbose: A boolean indicating whether we have to be verbose 
            (default is False).
        :type data_asdf: pandas.core.frame.DataFrame
        :type method: str among {'pearson', 'spearman', 'kendall'}
        :type perform_wgcna: bool
        :type estimate_beta: bool
        :type beta_value: int
        :type truncated: bool
        :type skip_first_value: bool
        :type beta_grid: numpy.ndarray(int)
        :type r2_threshold: float
        :type verbose: bool
        :return: The computed adjacency matrix (square).
        :rtype: numpy.ndarray(float)
    """

    # Get the number of entities to compute the correlation for.
    nb_entities = data_asdf.shape[1]

    # Get the indices of the upper triangle (diagonal excluded).
    data_uptri = np.triu_indices(nb_entities, 1, nb_entities)

    # We compute the correlation between all column pairwise.
    data_corr = data_asdf.corr(method)

    # We indicate that the correlation matrix has been computed.
    if verbose is True:
        print('Correlation matrix computed (using ' + str(method) + ').')

    # By default, the adjacency matrix is set to the correlation matrix, in
    # case WGCNA is not performed.
    data_adj = data_corr

    # We can perform WGCNA, which consists in rescaling the correlation [-1;1] 
    # to a similarity measure [0;1].
    if perform_wgcna is True:

        # By default, Beta is set to the parameter received.
        beta = beta_value

        # We rescale the correlation coefficients. 
        data_corr_min = np.min(np.min(data_corr))
        data_corr_max = np.max(np.max(data_corr))
        data_corr_std = (data_corr - data_corr_min) / (data_corr_max 
            - data_corr_min)

        # If we need to estimate Beta
        if estimate_beta is True:

            # We obtain the optimal beta through a grid search.
            (data_adj, beta, r2) = beta_gridsearch(data_corr_std, data_uptri, 
                truncated = truncated, skip_first_value = skip_first_value, 
                beta_grid = beta_grid, r2_threshold = r2_threshold)
        else:

            # We use the provided beta on the standardized correlations.
            data_adj = data_corr_std ** beta

        # We indicate that the adjacency matrix has been computed.
        if verbose is True:
            print('Adjacency matrix computed (Beta = ' + str(beta) + ').')

    # If the values are not within 0 and 1, we need to rescale them.
    # However, we do not always standardize the values, in some cases, we only 
    # rescale the left or the right border. 
    data_adj_min = np.min(np.min(data_adj))
    data_adj_max = np.max(np.max(data_adj))
    if data_adj_min < 0 and data_adj_max > 1:
        data_adj = (data_adj - data_adj_min) / (data_adj_max - data_adj_min)
    elif data_adj_min < 0: # We keep the max to 1.
        data_adj = (data_adj - data_adj_min) / (1 - data_adj_min)
    elif data_adj_max > 1: # We keep the min to 0.
        data_adj = data_adj / data_adj_max

    # For a correct adjacency matrix, we set the diagonal to zeros in all cases.
    np.fill_diagonal(data_adj.values, 0)

    # Return the computed adjacency matrix.
    return data_adj

# ============================================================================
#
#       FUNCTIONS - PLOT - SCATTER PLOT CORRELATION
#
# ============================================================================

def plot_scattercorrelations(first_plot_data, first_plot_data_tag, 
    second_plot_data, second_plot_data_tag, plot_filename, nb_bins = 0, 
    plot_title = '', verbose = False):

    """
        Build a comparison plot of two arrays.

        This function builds an image that contains two histograms of the 
        values from the two data arrays (left and right), and a scatter plot 
        between the two arrays (center). 

        :param first_plot_data: The first flatten array.
        :param first_plot_data_tag: The tag associated with the first array
            (used for the axis label).
        :param second_plot_data: The second flatten array.
        :param second_plot_data_tag: The tag associated with the second array
            (used for the axis label).
        :param plot_filename: The path to the file into which the plot is saved.
        :param nb_bins: The default number of bins for the two histograms 
            (default is 0, which let the program decides on the number of bins).
        :param plot_title: The title of the plot (default is '').
        :param verbose: A boolean indicating whether we have to be verbose 
            (default is False).
        :type first_plot_data: numpy.ndarray(float)
        :type first_plot_data_tag: str
        :type second_plot_data: numpy.ndarray(float)
        :type second_plot_data_tag: str
        :type plot_filename: str
        :type nb_bins: int
        :type plot_title: str
        :type verbose: bool
    """

    # We initialize the figure.
    scatter_figure = plt.figure(figsize = (15, 4))
    scatter_figure_gridspec = gridspec.GridSpec(1, 3)
    plt.title(plot_title)

    # First, we plot the histogram of the values of the first correlation 
    # dataset, on the right.
    mpp.plot_hist(scatter_figure_gridspec, 0, first_plot_data, 
        first_plot_data_tag, nb_bins = nb_bins)

    # Second, we create the scatter plot between the two correlation datasets
    # in the center.
    mpp.plot_scatter_flatarrays(scatter_figure_gridspec, 1, second_plot_data, 
        second_plot_data_tag, first_plot_data, first_plot_data_tag, 
        plot_title = plot_title)

    # Third, we plot the histogram of the values of the second correlation 
    # dataset, on the right.
    mpp.plot_hist(scatter_figure_gridspec, 2, second_plot_data, 
        second_plot_data_tag, nb_bins = nb_bins)

    # We configure the figure.
    plt.tight_layout()

    # We save the figure to a file.
    scatter_figure.savefig(plot_filename)

    # We close the figure (for the GC).
    plt.close()

    # We indicate that the figure has been created.
    if verbose is True:
        print('Correlation plot created (' + str(plot_filename) + ')')

# ============================================================================
#
#       FUNCTIONS - NETWORK CREATION
#
# ============================================================================

def create_network(data_adj, verbose = False):

    """
        Create a network from an adjacency matrix.

        This function creates a networkx object that represents a fully 
        connected weighted network. It is built from the provided adjacency 
        matrix that contains the weights of the edges.
 
        :param data_asdf: The square adjacency matrix.
        :param verbose: A boolean indicating whether we have to be verbose 
            (default is False).
        :type data_asdf: numpy.ndarray(float)
        :type verbose: bool
        :return: The built network.
        :rtype: networkx.classes.graph.Graph
    """

    # We create the networkx object from the provided adjacency matrix.
    data_network = nx.to_networkx_graph(data_adj)

    # We inform the user about the creation
    if verbose is True:
        print('Network created (with ' + str(data_network.number_of_nodes()) 
            + ' nodes and ' + str(data_network.number_of_edges()) + ' edges).')

    # We return the created network.
    return data_network

# ============================================================================
#
#       FUNCTIONS - WIPER - COMPUTE CENTRALITIES
#
# ============================================================================

def func_weighteddegree(data_network):

    """
        Compute the weighted degree of the network nodes.

        This function is an alias to the networkx method that computes the 
        weighted degree. It is useful in our settings to modularize the code 
        through the use of map and lambda.
 
        :param data_network: The network to consider.
        :type data_network: networkx.classes.graph.Graph
        :return: The weighted degree of all nodes within the provided network.
        :rtype: dict{int : float}
    """

    # We return the weighted degree as computed by networkx.
    return data_network.degree(weight = "weight")

def func_closenesscentrality(data_network):

    """
        Compute the closeness centrality of the network nodes.

        This function is an alias to the networkx method that computes the 
        closeness centrality. It is useful in our settings to modularize the 
        code through the use of map and lambda.
 
        :param data_network: The network to consider.
        :type data_network: networkx.classes.graph.Graph
        :return: The closeness centralities of all nodes within the provided 
            network.
        :rtype: dict{int : float}
    """

    # We return the normalized closeness centralities as computed by networkx.
    # Notice that the algorithm uses the edge weights in the process.
    return nx.closeness_centrality(data_network, distance = "weight", 
        normalized = True)

def func_betweennesscentrality(data_network):

    """
        Compute the betweenness centrality of the network nodes.

        This function is an alias to the networkx method that computes the 
        betweenness centrality. It is useful in our settings to modularize the 
        code through the use of map and lambda.
 
        :param data_network: The network to consider.
        :type data_network: networkx.classes.graph.Graph
        :return: The betweenness centralities of all nodes within the provided 
            network.
        :rtype: dict{int : float}
    """

    # We return the normalized betweenness centralities as computed by networkx.
    # Notice that the algorithm uses the edge weights in the process.
    return nx.betweenness_centrality(data_network, normalized = True, 
        weight = "weight")

def func_cfclosenesscentrality(data_network):

    """
        Compute the current flow closeness centrality of the network nodes.

        This function is an alias to the networkx method that computes the 
        current flow closeness centrality (aka, information centrality). It is 
        useful in our settings to modularize the code through the use of map 
        and lambda.
 
        :param data_network: The network to consider.
        :type data_network: networkx.classes.graph.Graph
        :return: The current flow closeness centralities of all nodes within 
            the provided network.
        :rtype: dict{int : float}
    """

    # We return the normalized current flow closeness centralities (aka, 
    # information centrality) as computed by networkx. Notice that the 
    # algorithm uses the edge weights in the process.
    return nx.current_flow_closeness_centrality(data_network, 
        weight = "weight", solver='lu')

def func_cfbetweennesscentrality(data_network):

    """
        Compute the current flow betweenness centrality of the network nodes.

        This function is an alias to the networkx method that computes the 
        current flow betweenness centrality (aka, random-walk betweenness 
        centrality). It is useful in our settings to modularize the code 
        through the use of map and lambda. When the network is too large, only
        an approximate of the metrics is computed.
 
        :param data_network: The network to consider.
        :type data_network: networkx.classes.graph.Graph
        :return: The current flow betweenness centralities of all nodes within 
            the provided network.
        :rtype: dict{int : float}
    """

    # We inititate the results.
    results = dict()

    # If the network is too large, we only compute an approximate.
    if data_network.number_of_nodes() > NB_NODES_APPROXIMATE_CFBC:
        results = nx.approximate_current_flow_betweenness_centrality(
            data_network, normalized = True, weight = 'weight', solver = 'lu')
    # Otherwise, we compute exact current flow betweenness centralities.
    else:
        results = nx.current_flow_betweenness_centrality(data_network, 
            normalized = True, weight = 'weight', solver = 'lu')

    # We return the normalized current flow betweenness centralities (aka, 
    # random-walk betweenness centrality) as computed by networkx.
    # Notice that the algorithm uses the edge weights in the process.
    return results

def func_eigenvectorcentrality(data_network):

    """
        Compute the eigenvector centrality of the network nodes.

        This function is an alias to the networkx method that computes the 
        eigenvector centrality. It is useful in our settings to modularize the 
        code through the use of map and lambda.
 
        :param data_network: The network to consider.
        :type data_network: networkx.classes.graph.Graph
        :return: The eigenvector centralities of all nodes within the provided 
            network.
        :rtype: dict{int : float}
    """

    # We return the eigenvector centralities as computed by networkx.
    # Notice that the algorithm uses the edge weights in the process.
    return nx.eigenvector_centrality_numpy(data_network, weight = "weight")

# TODO: we can define ourselves the alpha = 0.1 and beta = 1.0 values.
def func_katzcentrality(data_network):

    """
        Compute the katz centrality of the network nodes.

        This function is an alias to the networkx method that computes the 
        katz centrality. It is useful in our settings to modularize the 
        code through the use of map and lambda.
 
        :param data_network: The network to consider.
        :type data_network: networkx.classes.graph.Graph
        :return: The katz centralities of all nodes within the provided 
            network.
        :rtype: dict{int : float}
    """

    # We return the normalized katz centralities as computed by networkx.
    # Notice that the algorithm uses the edge weights in the process.
    # TODO: we can define ourselves the alpha = 0.1 and beta = 1.0 values.
    return nx.katz_centrality_numpy(data_network, normalized = True, 
        weight = "weight")

def func_loadcentrality(data_network):

    """
        Compute the load centrality of the network nodes.

        This function is an alias to the networkx method that computes the 
        load centrality. It is useful in our settings to modularize the 
        code through the use of map and lambda.
 
        :param data_network: The network to consider.
        :type data_network: networkx.classes.graph.Graph
        :return: The load centralities of all nodes within the provided 
            network.
        :rtype: dict{int : float}
    """

    # We return the normalized load centralities as computed by networkx.
    # Notice that the algorithm uses the edge weights in the process.
    return nx.load_centrality(data_network, normalized = True, 
        weight = "weight")

# Possible methods are 'func_weighteddegree', 'degree_centrality', 
#   'func_closenesscentrality', 'func_betweennesscentrality', 
#   'func_cfclosenesscentrality', 'func_cfbetweennesscentrality', 
#    'func_eigenvectorcentrality', 'func_katzcentrality', 
#    'func_loadcentrality', 'communicability' / 'communicability_exp'.
# TODO : add multiple methods to do the randomization only once.
def get_centrality_measures(data_adj, data_network, centrality_method, 
    compute_stats = True, nb_runs = 1000, verbose = False):

    """
        Compute node centralities and the associated statistics.

        This function first computes some centrality values for all nodes in 
        the provided network, using the provided function. Then, the function
        randomizes the network several times to assess the significance of the 
        observed values using the FDR method. The adjacency matrix and the 
        network must always correspond to the same data since the randomization 
        is using the adjacency matrix while the centrality measures are 
        computed on the network.

        The computation of the significance is still subject to debate and 
        tests. The current solution is to use a gaussian kernel to fit all the 
        centrality values obtained for all nodes and all repetitions, and use 
        this fit to compute the p-values. This is only valid when working on 
        fully connected networks.
 
        :param data_adj: The adjacency matrix of the network.
        :param data_network: The network that corresponds to the matrix.
        :param centrality_method: The method to use to compute the centrality 
            values (can be a user based function).
        :param compute_stats: A boolean indicating whether we have to compute 
            statistics in addition to the centrality values (default is True).
        :param nb_runs: The number of randomizations used to estimate the FDR 
            p-values (default is 1000).
        :param verbose: A boolean indicating whether we have to be verbose 
            (default is False).
        :type data_adj: numpy.ndarray(float)
        :type data_network: networkx.classes.graph.Graph
        :type centrality_method: function (networkx.classes.graph.Graph(float)
            : dict {int :_ float})
        :type compute_stats: bool
        :type nb_runs: int
        :type verbose: bool
        :return: The observed centrality values and the associated FDR values.
        :rtype: (numpy.ndarray(float), numpy.ndarray(float))
    """

    # We compute the centrality values on the network. These represent the 
    # observed values.
    data_cents = list(map(lambda x : x(data_network), [centrality_method]))[0]
    data_cents_asarray = np.array(list(data_cents.values()))

    # If we do not have to compute statistics, we can stop here, else
    # we go on with the computation. 
    if compute_stats is False:
        return data_cents_asarray, None, None

    # We compute the indices of the upper and lower triangles 
    # (diagonal excluded). This is helpful for the randomization.
    nb_nodes = data_network.number_of_nodes()
    data_uptri = np.triu_indices(nb_nodes, 1, nb_nodes)
    data_lotri = np.tril_indices(nb_nodes, -1, nb_nodes)

    # We initialize the vectors that contain the statistics.
    data_rawfdr = np.zeros(nb_nodes)
    data_defaultfdr = np.zeros(nb_nodes) + 1

    # We initialize the array that will contain all centrality values from the 
    # randomized networks. It will be used for the computation of the 
    # significance.
    all_rand_cents = np.zeros([nb_nodes, nb_runs + 1])
    all_rand_cents[:,nb_runs] = data_cents_asarray
    # all_rand_cents = np.zeros([nb_nodes, nb_runs])

    # For each randomization, we compute the centrality values again and keep 
    # track of the number of cases for which we obtain at least as good values 
    # as the real observed values.
    for i in np.arange(nb_runs):

        # We start by creating the random network. Here we simply randomize the 
        # upper triangle of the adjacency matrix and then update the lower 
        # triangle accordingly (to keep the adjacency matrix symmetric).
        # We start by permuting the upper triangle.
        rand_adj = data_adj
        rand_adj[data_uptri] = np.random.permutation(data_adj[data_uptri])

        # We then transpose the randomized upper triangle and sum up the two to 
        # get the symmetric randomized adjacency matrix.
        data_adj[data_lotri] = 0
        rand_adj = rand_adj + rand_adj.transpose()

        # We create a network from that random adjacency matrix.
        rand_network = nx.to_networkx_graph(rand_adj)

        # We compute the centrality values associated to this randomized 
        # network.
        rand_cents = list(map(lambda x : x(rand_network), 
            [centrality_method]))[0]
        rand_cents_asarray = np.array(list(rand_cents.values()))

        # Store the current centralities.
        all_rand_cents[:,i] = rand_cents_asarray

        # We update the FDR when the score is better than or equal to the 
        # observed score. 
        update = rand_cents_asarray >= data_cents_asarray
        data_rawfdr[update] += 1

        # We print an update about the status once in a while...
        if verbose is True:
            if i % 100 == 99:
                print(str(i + 1) + " permutations done.")

    # End for i in np.arange(nb_runs):

    # After all runs are done, we compute the real FDR values that are already 
    # tested for multiple testing.
    data_rawfdr = (data_rawfdr / nb_runs) * nb_nodes

    # Since the multiple testing correction might results in FDR > 1, we 
    # automatically set these to 1.
    data_fdr = np.minimum(data_rawfdr, data_defaultfdr)

    # We prepare the data for the Gaussian kernel fit. 
    all_rand_cents_to_fit = all_rand_cents.flatten()
    fitdata_min = np.min([np.min(data_cents_asarray), 
        np.min(all_rand_cents_to_fit)])
    fitdata_max = np.max([np.max(data_cents_asarray), 
        np.max(all_rand_cents_to_fit)])
    fitdata_range = (fitdata_max - fitdata_min)
    fitdata_min = fitdata_min - fitdata_range / 3
    fitdata_max = fitdata_max + fitdata_range / 3

    # # We print the data for DEBUG
    # df = p.DataFrame(all_rand_cents_to_fit)
    # print("[DBG] We print a summary of the data to be fit:")
    # print(df.describe())
    # print("[DBG] DF shape is ")
    # print(df.shape)
    # print("[DBG] nb null/infinites is ")
    # df.replace([np.inf, -np.inf], np.nan)
    # print(df.isnull().sum())
    # print("[DBG] Max - min  is ")
    # print(str(np.max(np.max(df)) - np.min(np.min(df))))

    # df_min = np.min(np.min(all_rand_cents_to_fit))
    # df_max = np.max(np.max(all_rand_cents_to_fit))
    # all_rand_cents_to_fit2 = (all_rand_cents_to_fit - df_min) / (df_max - df_min)

    # print("[DBG2] We print a summary of the data to be fit:")
    # print(df2.describe())
    # print("[DBG2] DF shape is ")
    # print(df2.shape)
    # print("[DBG2] nb null/infinites is ")
    # print(df2.isnull().sum())
    # print("[DBG2] Max - min  is ")
    # print(str(np.max(np.max(df2)) - np.min(np.min(df2))))

    # We fit a Gaussian kernel to the data.
    rand_cent_gkdefit = stats.gaussian_kde(all_rand_cents_to_fit)

    # We check that the fit did indeed cover the whole dataset (sanity check)
    if rand_cent_gkdefit.integrate_box_1d(fitdata_min, fitdata_max) != 1:
        print("[Warning] Problem with the Gaussian kernel fit (function does "
            + "not integrate to 1).")

    # We compute p-values based on the fit.
    data_fit_rawfdr = np.array([rand_cent_gkdefit.integrate_box_1d(
        observed_value, fitdata_max) for observed_value in data_cents_asarray])
    data_fit_rawfdr = data_fit_rawfdr * nb_nodes
    data_fit_fdr = np.minimum(data_fit_rawfdr, data_defaultfdr)

    # We also replace the 0 FDR values by the actual minimum.
    if len(data_fit_fdr[data_fit_fdr == 0]) > 0:
        data_fit_fdr[data_fit_fdr == 0] = np.min(data_fit_fdr[data_fit_fdr != 0])

    # We return the observed centrality values and the corresponding FDR values.
    return data_cents_asarray, data_fdr, data_fit_fdr

# ============================================================================
#
#       FUNCTIONS - PLOT - CENTRALITIES
#
# ============================================================================

def plot_centralitydistribution(data_asarray, plot_filename, nb_bins = 0, 
    plot_title = '', verbose = False):

    """
        Plot the distribution of the centrality values.

        This function plots the value distribution of a data array that 
        contains centrality measures, and saves the figure to a file.
 
        :param data_asarray: The array that contains the centrality values.
        :param plot_filename: The path to the filename used to save the image.
        :param nb_bins: The number of bins for the histogram (default is 0, 
            which let the program decides on the optimal number of bins).
        :param plot_title: The title of the plot (default is '').
        :param verbose: A boolean indicating whether we have to be verbose 
            (default is False).
        :type data_asarray: numpy.ndarray(float)
        :type plot_filename: str
        :type nb_bins: int
        :type plot_title: str
        :type verbose: bool
    """

    # We initialize the figure.
    dist_figure = plt.figure(figsize = (6, 6))
    dist_figure_gridspec = gridspec.GridSpec(2, 1, height_ratios = [1, 4])

    # First, we build a violin-plot on top.
    plt.subplot(dist_figure_gridspec[0])
    plt.axis('off')
    plt.violinplot(data_asarray, vert = False)
    plt.title(plot_title)

    # Second, we plot the histogram of the values at the bottom.
    mpp.plot_hist(dist_figure_gridspec, 1, data_asarray, '', nb_bins = nb_bins)

    # We configure the figure.
    plt.tight_layout()

    # We save the figure to a file.
    dist_figure.savefig(plot_filename)

    # We close the figure (for the GC).
    plt.close()

    # We indicate that the figure has been created.
    if verbose is True:
        print('Centrality distribution created (' + str(plot_filename) + ')')

# ============================================================================
#
#       FUNCTIONS - I/OS - WRITE TO FILES
#
# ============================================================================

def save_centralities(centrality_values, centrality_stats, filename, 
    index_values = [], centrality_tag = '', sep = '\t', header = False, 
    index = False):

    """
        Save the centrality values and statistics to a file.

        This function saves centrality values and the associated statistics to 
        a file using the TSV format by default. The function can be configured 
        to use any variation of the CSV format.
 
        :param centrality_values: The centrality measures that will be saved 
            into the TSV file.
        :param centrality_stats: The statistics associated with the centrality 
            measures that will be saved into the file as well.
        :param filename: The path to the filename used to save the data.
        :param index_values: The optional row indexes (default is an empty 
            array).
        :param centrality_tag: The tag to use to name the columns (default 
            is '').
        :param sep: The separator to use when saving the data as TSV (default 
            is 'tab').
        :param header: A boolean indicating whether we should print a header 
            row (default is False).
        :param index: A boolean indicating whether we should print the row 
            indexes (default is False).
        :type centrality_values: numpy.ndarray(float)
        :type centrality_stats: numpy.ndarray(float)
        :type filename: str
        :type index_values: numpy.ndarray(str)
        :type centrality_tag: str
        :type sep: str
        :type header: bool
        :type index: bool
    """

    # We initiate the dataframe column names. 
    col1_name = centrality_tag + '_values'
    col2_name = centrality_tag + '_stats'

    # If we only have the values, and not the stats.
    if centrality_stats is None:
        print_data_values = {col1_name: centrality_values}
        print_data = p.DataFrame(print_data_values, index = index_values, 
            columns = [col1_name])

    # We have both the values and the stats.
    else:
        print_data_values = {col1_name: centrality_values, 
            col2_name: centrality_stats}
        print_data = p.DataFrame(print_data_values, index = index_values, 
            columns = [col1_name, col2_name])

    # We save it to a file as a TSV file.
    print_data.sort_index(inplace = True)
    print_data.to_csv(path_or_buf = filename, sep = sep, header = header, 
        index = index)
