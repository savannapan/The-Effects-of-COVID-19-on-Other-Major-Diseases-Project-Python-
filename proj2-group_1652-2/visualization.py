""""Description"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from load_data import get_dataset_values, group_datasets
from correlation import pearson
from linear_reg import retrieve_data, generate_best_fit_line

DATA = get_dataset_values()

###################################################################################################
# Visualization for regression line (Scatterplot + regression line)
###################################################################################################


def plot_regression_line() -> None:
    """function that plots the scatter plot and regression line of the given data using the inputs
    hosp, death_counts and coeff which is a tuple consisting of the slope and y-int of the line.
    """
    hosp, death_counts = retrieve_data()
    coeff = generate_best_fit_line()
    # creating a scatter plot for the given points
    plt.scatter(hosp, death_counts, color="g", marker="s", s=50)

    # predicted death counts
    y_predicted = (coeff[0] * hosp) + coeff[1]

    # plot the linear regression line
    plt.plot(hosp, y_predicted, color="b")

    # labelling the x and y axis
    plt.xlabel('Hospitalizations')
    plt.ylabel('Total Death Counts')

    plt.show()


###################################################################################################
# Visualization for correlations (Heatmap)
###################################################################################################


def plot_correlations() -> None:
    """Yes"""
    correlations = get_correlations()
    labels = list(group_datasets().columns)

    row = labels
    column = labels

    data = pd.DataFrame(correlations, columns=column, index=row)

    plt.xticks([0, 1, 2, 3, 4], labels=column)
    plt.yticks([0, 1, 2, 3, 4], labels=row)
    # vertically rotating the ticks of the x axis
    plt.xticks([0, 1, 2, 3, 4], row, rotation='vertical')

    # Display dataframe in the form of a heatmap with different shades of green
    plt.imshow(data, cmap="Greens")

    plt.colorbar()
    plt.show()


def get_correlations() -> np.array:
    """Return an array of a list of lists, where each inner list contains the values of the
     correlation coefficients for the relationship between each variable and other variables
     in the imported dataset.
     """
    # ACCUMULATOR outer_lst: keeps track of the inner lists
    outer_lst = []

    for variable_v1 in DATA:
        # ACCUMULATOR inner_lst: keeps track of the correlation
        # coefficients between a variable and the others
        inner_lst = []

        for variable_v2 in DATA:
            value = pearson(variable_v1, variable_v2)
            inner_lst.append(value)

        outer_lst.append(inner_lst.copy())

    return np.array(outer_lst)


if __name__ == '__main__':
    import python_ta
    import python_ta.contracts

    python_ta.contracts.DEBUG_CONTRACTS = False
    python_ta.contracts.check_all_contracts()

    python_ta.check_all(config={
        'allowed-io': [],
        'extra-imports': ['python_ta.contracts', 'load_data', 'correlation', 'linear_reg',
                          'matplotlib.pyplot', 'pandas', 'numpy'],
        'max-line-length': 100,
        'disable': ['R1705', 'C0200'],
    }, output='pyta_report.html')
