import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

from survey_data_dictionary import DATA_DICTIONARY

# Load data

# We want to take the names list from our data dictionary
names = [x.name for x in DATA_DICTIONARY]

# Generate the list of names to import
usecols = [x.name for x in DATA_DICTIONARY if x.usecol]

# dtypes should be a dict of 'col_name' : dtype
dtypes = {x.name : x.dtype for x in DATA_DICTIONARY if x.dtype}

# same for converters
converters = {x.name : x.converter for x in DATA_DICTIONARY if x.converter}

df = pd.read_csv('data/python_psf_external.csv',
                 header=0,
                 names=names,
                 dtype=dtypes,
                 converters=converters,
                 usecols=usecols)

#%% Clean up data: remove disqualified users

# In the survey, any user who selected they don't use Python was then
# disqualified from the rest of the survey. So let's drop them here.
df = df[df['python_main'] != 'No, I don’t use Python for my current projects']

# Considering we now only have two categories left:
#  - Yes
#  - No, I use Python for secondary projects only
# Let's turn it into a bool

df['python_main'] = df['python_main'] == 'Yes'

#%% Plot the web dev / data scientist ratio

# In the survey, respondents were asked to estimate the ratio between
# the amount of web developers vs the amount of data scientists. Afterwards
# they were asked what they thought the most popular answer would be.
# Let's see if there's a difference!

# This is a categorical data point, and it's already ordered in the data
# dictionary. So we shouldn't sort it after counting the values.
ratio_self = df['webdev_science_ratio_self'].value_counts(sort=False)
ratio_others = df['webdev_science_ratio_others'].value_counts(sort=False)

# Let's draw a bar chart comparing the distributions
fig = plt.figure()
ax = fig.add_subplot(111)

RATIO_COUNT = ratio_self.count()
x = np.arange(RATIO_COUNT)
WIDTH = 0.4

self_bars = ax.bar(x, ratio_self, width=WIDTH, color='b', align='edge')
others_bars = ax.bar(x + WIDTH, ratio_others, width=WIDTH, color='g', align='edge')

# With these changes, no part of any bar has a negative x coordinate

ax.set_xlabel('Ratios')
ax.set_ylabel('Observations')
labels = [str(x) for x in ratio_self.index]
ax.set_xticklabels(labels)

# This creates the following xticks (ax.xaxis.majorticks):
# [Text(-1,0,'10:1'), Text(0,0,'5:1'), Text(1,0,'2:1'), Text(2,0,'1:1'),
# Text(3,0,'1:2'), Text(4,0,'1:5'), Text(5,0,'1:10'), Text(6,0,''),
# Text(7,0,''), Text(8,0,'')]

ax.legend((self_bars[0], others_bars[0]),
          ('Self', 'Most popular'))

plt.show()

#%% Calculate the predicted totals

# Let's recode the ratios to numbers, and calculate the means
CONVERSION = {
    '10:1': 10,
    '5:1' : 5,
    '2:1' : 2,
    '1:1' : 1,
    '1:2' : 0.5,
    '1:5' : 0.2,
    '1:10': 0.1
}

self_numeric = df['webdev_science_ratio_self'] \
                    .replace(CONVERSION.keys(), CONVERSION.values())

others_numeric = df['webdev_science_ratio_others'] \
                    .replace(CONVERSION.keys(), CONVERSION.values())

print(f'Self:\t\t{self_numeric.mean().round(2)} web devs / scientist')
print(f'Others:\t\t{others_numeric.mean().round(2)} web devs / scientist')

#%% Is the difference statistically significant?

result = scipy.stats.chisquare(ratio_self, ratio_others)

# The null hypothesis is that they're the same. Let's see if we can reject it

print(result)
