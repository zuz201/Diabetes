import numpy as np
import pandas as pd
import seaborn as sns


from functions import plot_outliers

dataset = pd.read_csv('data/diabetes.txt')

dataset.info()

dataset.info()

dataset.groupby('Outcome').size()
sns.countplot(dataset['Outcome'])

dataset.head()
description = dataset.describe()


plot_outliers(dataset)
