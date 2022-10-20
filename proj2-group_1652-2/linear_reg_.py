""""Description"""

import matplotlib.pyplot as plt
from load_data import group_datasets


def retrieve_data() -> tuple:
    """retrieve data"""
    data = group_datasets()

    # collecting hospitalization data and total death counts
    hosp = data['Hospitalizations'].values
    sum_death_counts = data['Sum of Death Counts'].values

    return hosp, sum_death_counts


def generate_best_fit_line() -> tuple:
    """Calculate the line of best fit"""
    hosp, death_counts = retrieve_data()

    # calculating mean of hospitalization and total death counts
    mean_hosp = sum(hosp) / len(hosp)
    mean_death_count = sum(death_counts) / len(death_counts)

    total_values = len(hosp)

    # calculating slope and y-int of the line
    num = 0
    den = 0

    for i in range(0, total_values):
        num = num + ((hosp[i] - mean_hosp) * (death_counts[i] - mean_death_count))
        den = den + (hosp[i] - mean_hosp) ** 2

    slope = num / den
    y_int = mean_death_count - (slope * mean_hosp)
    coeff = (slope, y_int)

    return coeff


def get_regression_line() -> None:
    """function that plots the scatter plot and regression line of the given data
    """
    hosp, death_counts = retrieve_data()
    coeff = generate_best_fit_line()
    plt.xlabel('Hospitalizations')
    plt.ylabel('Total Death Counts')

    # creating a scatter plot for the given points
    plt.scatter(hosp, death_counts, color="g", marker="s", s=50)

    # predicted death counts
    death_predicted = (coeff[0] * hosp) + coeff[1]

    # plot the linear regression line
    plt.plot(hosp, death_predicted, color="b")

    plt.show()


if __name__ == '__main__':
    import python_ta
    import python_ta.contracts

    python_ta.contracts.DEBUG_CONTRACTS = False
    python_ta.contracts.check_all_contracts()

    python_ta.check_all(config={
        'allowed-io': [],
        'extra-imports': ['python_ta.contracts', 'matplotlib.pyplot', 'load_data'],
        'max-line-length': 100,
        'disable': ['R1705', 'C0200'],
    }, output='pyta_report.html')
