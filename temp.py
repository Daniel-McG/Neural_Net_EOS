import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import numpy as np

# Dummy data
categories = ['Category 1', 'Category 2']
group1_values = [20, 35]
group2_values = [15, 25]

# Set up positions for the bars
bar_width = 0.35
index = np.arange(len(categories))
bar_positions_group1 = index - bar_width / 2
bar_positions_group2 = index + bar_width / 2

# Create the bar plot
plt.bar(bar_positions_group1, group1_values, bar_width, label='Group 1')
plt.bar(bar_positions_group2, group2_values, bar_width, label='Group 2')

# Add labels and title
plt.xlabel('Categories')
plt.ylabel('Values')
plt.title('Bar Plot with Two Groups')
plt.xticks(index, categories)
plt.legend()

# Show the plot
plt.show()
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import numpy as np

# Dummy data
categories = ['Category 1', 'Category 2']
group1_values = [20, 35]
group2_values = [15, 25]

# Set up positions for the bars
bar_width = 0.35
index = np.arange(len(categories))
bar_positions_group1 = index - bar_width / 2
bar_positions_group2 = index + bar_width / 2
fig,axis = plt.subplots(nrows=1,ncols=2)
# Create the bar plot
axis[0].bar(bar_positions_group1, group1_values, bar_width, label='Group 1')
axis[1].bar(bar_positions_group2, group2_values, bar_width, label='Group 2')

# Add labels and title
axis[0].set_xlabel('Categories')
axis[0].set_ylabel('Values')
axis[0].set_title('Bar Plot with Two Groups')
axis[0].set_xticks(index, categories)

axis[1].set_xlabel('Categories')
axis[1].set_ylabel('Values')
axis[1].set_title('Bar Plot with Two Groups')
axis[1].set_xticks(index, categories)

# Show the plot
plt.show()
