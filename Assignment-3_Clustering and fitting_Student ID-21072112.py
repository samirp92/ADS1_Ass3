# Import the required libraries
import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt  
import seaborn as sns  
from sklearn.cluster import KMeans
import sklearn.metrics as metrics
import sklearn.cluster as cluster
import sklearn.metrics as skmet
import scipy.optimize as opt
import err_ranges as err

def read_data(file_name):  # Defind a fuction to read the files
    df = pd.read_csv(file_name, header = [3]) # To read the file
    new_col_name = df.iloc[0,2]  # Store the indicator name in new variable
    df = df.iloc[:,[0,64]]  # Remove unnecessary columns
    df.columns = ['Country Name', new_col_name] # Set the columns name
    print(df)  # To print the data
    return df

data1 = read_data('Urban population (% of total population).csv')  # To read & store data-1
data2 = read_data('Labor force, total.csv')  # To read & store data-2
data3 = read_data('GDP growth (annual %).csv')  # To read & store data-3
data4 = read_data('Tax revenue (% of GDP).csv')  # To read & store data-4

# Merge all data frames
data = pd.merge(data1, data2, on = 'Country Name').merge(data3, on = 'Country Name').merge(data4, on = 'Country Name')
print(data)
data.to_excel('data.xlsx')

# Define functions to normalise one array and iterate over all numerical columns of the dataframe
def norm(array): # Returns array normalised to [0,1]. Array can be a numpy array or a column of a data frame
    min_val = np.min(array)
    max_val = np.max(array)
    scaled = (array-min_val) / (max_val-min_val)
    return scaled

def norm_df(df, first=0, last=None):
    # iterate over all numerical columns
    for col in df.columns[first:last]: # excluding the first column
        df[col] = norm(df[col])
    return df

# Define the fuction for heat map to find the co-relation between indicators
def map_corr(df, size=4):
    plt.figure(dpi =300)
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(size, size))
    ax.matshow(corr, cmap='coolwarm')
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)

corr = data.corr()
map_corr(data)
plt.show()

# The scatter plot to ensure the best co-relation ontained from heat map
plt.figure(dpi =300)
pd.plotting.scatter_matrix(data, figsize=(10,10), color = 'b')
plt.tight_layout() # helps to avoid overlap of labels
plt.show()

# Create the new data frame with proper co-relation between indicators
dataframe = pd.merge(data1, data3, on = 'Country Name')  # To merge the two data frame
dataframe = dataframe.dropna().reset_index(drop = True)  # Drop NaN values
dataframe.to_excel('dataframe.xlsx')  # To store the data frame
features = ['Urban population (% of total population)', 'GDP growth (annual %)']  # To filter the value of selected indicator only
df = dataframe[features].copy()  # To store the value of selected indicator only
df.to_excel('df.xlsx')  # To save the data
df = norm_df(df)  # Convert the data into normalized form
X = df.values  # To store the data as values