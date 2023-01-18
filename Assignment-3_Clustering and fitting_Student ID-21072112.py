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

# Find the WCSS to find the correct cluster number for kmeans clustering
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state = 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
    
#  Plot an elbow graph
plt.figure(dpi =300)
plt.plot(range(1,11), wcss)
plt.xlabel('Number of Clusters', fontsize = 12)
plt.ylabel('WCSS', fontsize = 12)
plt.title('The Elbow Point Graph', fontsize = 15)
plt.show()

# To find the silhouette_score for number of cluster
print('Silhouette score')
for ic in range(2, 7):
    # set up kmeans and fit
    kmeans = cluster.KMeans(n_clusters=ic)
    kmeans.fit(X)
    # extract labels and calculate silhoutte score
    labels = kmeans.labels_
    print(ic, skmet.silhouette_score(X, labels))
   
kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 0)

# Return a label for each data point based on their cluster
Y = kmeans.fit_predict(X)

# Extract labels and cluster centres
labels = kmeans.labels_
cen = kmeans.cluster_centers_

# plotting all the clusters and their Centroids
plt.figure(figsize=(8,8), dpi =300)
plt.scatter(X[Y==0,0], X[Y==0,1], s = 75, c = 'green', label = 'Cluster 0')
plt.scatter(X[Y==1,0], X[Y==1,1], s = 75, c = 'red', label = 'Cluster 1')
plt.scatter(X[Y==2,0], X[Y==2,1], s = 75, c = 'yellow', label = 'Cluster 2')

# Plot the centroids
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s = 100, c = 'blue', label = 'Centroids')
plt.title('Clusters (Urban population vs GDP growth)', fontsize = 20)
plt.xlabel('Urban population (% of total population) in 2020', fontsize = 18)
plt.ylabel('GDP growth (annual %) in 2020', fontsize = 18)
plt.legend(bbox_to_anchor = (1.0, 1.0), fontsize = 18, loc = 'upper left')
plt.show()

# Extract & Store the group of country according to labels
country_group0 = dataframe[labels == 0]
country_group1 = dataframe[labels == 1]
country_group2 = dataframe[labels == 2]

country_group0.to_excel('country_group0.xlsx')
country_group1.to_excel('country_group1.xlsx')
country_group2.to_excel('country_group2.xlsx')
