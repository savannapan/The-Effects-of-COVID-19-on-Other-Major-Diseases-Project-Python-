""""Description"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from load_data import get_dataset_values
from load_data import group_datasets
from correlation import pearson


DATA = get_dataset_values()


def get_correlations() -> np.array:
    """generate array of correlations to display"""
    outer_lst = []
    for variable_v1 in DATA:
        inner_lst = []
        for variable_v2 in DATA:
            value = pearson(variable_v1, variable_v2)
            inner_lst.append(value)
        outer_lst.append(inner_lst.copy())

    return np.array(outer_lst)


def plot_correlations() -> None:
    """Yes"""
    correlations = get_correlations()
    labels = list(group_datasets().columns)


row = ['sum of death counts', 'Malignant Neoplasms d.c.', 'Diseases of the Heart d.c',
       'Chronic lower respiratory diseases', 'Hospitalizations']

column = ['sum of death counts', 'Malignant Neoplasms d.c.', 'Diseases of the Heart d.c',
          'Chronic lower respiratory diseases', 'Hospitalizations']

data = pd.DataFrame(get_correlations(), columns=column, index=row)


plt.xticks([0, 1, 2, 3, 4], labels=column)
plt.yticks([0, 1, 2, 3, 4], labels=row)
# vertically rotating the ticks of the x axis
plt.xticks([0, 1, 2, 3, 4], row, rotation='vertical')


# Display dataframe in the form of a heatmap with different shades of green
plt.imshow(data, cmap="Greens")

plt.colorbar()
plt.show()
