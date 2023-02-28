# Alaa Mohamed Galal Eldin
# 19P4206

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import MinMaxScaler

DataFrame = pd.DataFrame({'Region': ['India', 'Brazil', 'USA', 'Brazil', 'USA', 'India', 'Brazil', 'India', 'USA', 'India'],
                          'Age': [49, 32, 35, 43, 45, 40, np.nan, 53, 55, 42],
                          'Income': [86400, 57600, 64800, 73200, np.nan, 69600, 62400, 94800, 99600, 80400],
                          'Online Shopper': ['No', 'Yes', 'No', 'No', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes']})

# Excluding data that is not numerical from data frame
Numerical_Features = DataFrame.select_dtypes(exclude=['object']).columns.tolist()
print(Numerical_Features)
DataFrame_N = DataFrame[Numerical_Features]
print(DataFrame_N)

# Handling missing numerical data by using mean
Imputer_Mean = SimpleImputer(missing_values=np.nan, strategy='mean')
Imputer_Mean.fit(DataFrame_N)
DataFrame_N = Imputer_Mean.transform(DataFrame_N)
print(DataFrame_N)

# Putting numerical data in data frame after data cleaning
DataFrame[Numerical_Features] = DataFrame_N


# Discretization of numerical data into 3 bins using uniform strategy and ordinal encoding
Categorize_Features = DataFrame.select_dtypes(exclude=['object']).columns.tolist()
DataFrame_Categorize = DataFrame[Categorize_Features]
est = KBinsDiscretizer(n_bins=3, strategy='uniform', encode='ordinal')
c = est.fit(DataFrame_Categorize)
print(c.bin_edges_)
DataFrame_Categorize = est.transform(DataFrame_Categorize)
print(DataFrame_Categorize)

# Normalization of numerical data into min/max scaler using range (0,1)
Normalizer = MinMaxScaler(feature_range=(0, 1))
Norm_Data = Normalizer.fit_transform(DataFrame_N)
print(Norm_Data)

# Putting all data in data frame after data transformation
DataFrame[Numerical_Features] = Norm_Data
print(DataFrame)