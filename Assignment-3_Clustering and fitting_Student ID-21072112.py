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
# Data preparation for curve fitting
def read_data(file_name, country, save_file):  # Defind a fuction to read the files
    df = pd.read_csv(file_name, header = [3]) # To read the file
    #df = df[df['Indicator Name'].isin([indicator])].reset_index(drop = True)  # To filter the data with required indicator
    df = df[df['Country Name'].isin([country])].reset_index(drop = True)  # To filter the data with required country
    df = df.T.reset_index(drop = False)  # To transpose the data
    new_col1 = 'Year' 
    new_col2 = df.iloc[2,1]
    df.columns = [new_col1, new_col2]  # Set the columns name
    df = df.iloc[4:,:].reset_index(drop = True)  # To remove unnecessary rows
    df = df.dropna().reset_index(drop = True)  # To drop NaN values
    df = df.astype(float)  # To convert the data into float
    print(df)  # To print the data
    df.to_excel(save_file)  # To save the file
    return df

df1 = read_data('Urban population (% of total population).csv', 'Iran, Islamic Rep.', 'df1.xlsx')  # To call the fuction & store in df1
df2 = read_data('Labor force, total.csv', 'Iran, Islamic Rep.', 'df2.xlsx')  # To call the fuction & store in df2
df3 = read_data('Urban population (% of total population).csv', 'Ireland', 'df3.xlsx')  # To call the fuction & store in df3
df4 = read_data('Labor force, total.csv', 'Ireland', 'df4.xlsx')  # To call the fuction & store in df4
df5 = read_data('Urban population (% of total population).csv', 'Egypt, Arab Rep.', 'df5.xlsx')  # To call the fuction & store in df3
df6 = read_data('Labor force, total.csv', 'Egypt, Arab Rep.', 'df6.xlsx')  # To call the fuction & store in df4

# Define the exponential function and the logistics functions for fitting
def exp_growth(t, scale, growth):  # Define exponential growth function
    f = scale * np.exp(growth * (t-1960)) 
    return f
       
def logistics(t, scale, growth, t0):  # Define logistic function
    f = scale / (1.0 + np.exp(-growth * (t - t0)))
    return f

def err_ranges(x, func, param, sigma):  # Define the function to calculates the upper and lower limits
    import itertools as iter
    
    # initiate arrays for lower and upper limits
    lower = func(x, *param)
    upper = lower
    
    uplow = []   # list to hold upper and lower limits for parameters
    for p,s in zip(param, sigma):
        pmin = p - s
        pmax = p + s
        uplow.append((pmin, pmax))
        
    pmix = list(iter.product(*uplow))
    
    for p in pmix:
        y = func(x, *p)
        lower = np.minimum(lower, y)
        upper = np.maximum(upper, y)
        
    return lower, upper 

# Fit the logistic fuction for df1
param, covar = opt.curve_fit(logistics, df1['Year'], df1['Urban population (% of total population)'], p0 = (2e9, 0.05, 1960.0), maxfev = 1000)
df1['fit'] = logistics(df1['Year'], *param)

sigma = np.sqrt(np.diag(covar))

# Set upper and lower limit for df1
low, up = err.err_ranges(df1["Year"], logistics, param, sigma)

# Plot the graph with original data & logistic fuction data with lower & upper limit for df1
plt.figure(dpi = 300)  # To create figure with required resolution 
plt.plot(df1['Year'], df1['Urban population (% of total population)'], label = "Iran's urban population (%)") # To plot line graph with actual data
plt.plot(df1['Year'], df1['fit'], label = 'fit')  # To plot the 
plt.fill_between(df1['Year'], low, up, alpha = 0.7, color = 'green')
plt.xlabel('Year', fontsize = 12)
plt.ylabel("Iran's urban population (%)", fontsize = 12)
plt.title("Iran's urban population (%) with logistic fuction", fontsize = 15)
plt.legend()
plt.show()

# Forcast the Urban population for upcomming years with lower & upper limit for df1
print("Forcasted Iran's urban population (%)")
low, up = err.err_ranges(2030, logistics, param, sigma)
mean = (up+low) / 2.0
pm = (up-low) / 2.0
print("2030:", round(mean, 3), "+/-", round(pm, 3))
low, up = err.err_ranges(2040, logistics, param, sigma)
mean = (up+low) / 2.0
pm = (up-low) / 2.0
print("2040:", round(mean, 3), "+/-", round(pm, 3))

# Fit the logistic fuction for Total labor force for df2
param, covar = opt.curve_fit(logistics, df2['Year'], df2['Labor force, total'], p0 = (2e9, 0.05, 1960.0), maxfev = 1000)
df2['fit'] = logistics(df2['Year'], *param)

sigma = np.sqrt(np.diag(covar))

# Set upper and lower limit for df2
low, up = err.err_ranges(df2["Year"], logistics, param, sigma)

# Plot the graph with original data & logistic fuction data with lower & upper limit for df2
plt.figure(dpi = 300)  # To create figure with required resolution 
plt.plot(df2['Year'], df2['Labor force, total'], label = "Iran's total Labor force") # To plot line graph with actual data
plt.plot(df2['Year'], df2['fit'], label = 'fit')  # To plot the 
plt.fill_between(df2['Year'], low, up, alpha = 0.7, color = 'green')
plt.xlabel('Year', fontsize = 12)
plt.ylabel("Iran's total Labor force", fontsize = 12)
plt.title("Iran's total Labor force with logistic fuction", fontsize = 15)
plt.legend()
plt.show()

# Forcast the labor force for upcomming years with lower & upper limit for df2
print("Forcasted Iran's total Labor force")
low, up = err.err_ranges(2030, logistics, param, sigma)
mean = (up+low) / 2.0
pm = (up-low) / 2.0
print("2030:", round(mean, 3), "+/-", round(pm, 3))
low, up = err.err_ranges(2040, logistics, param, sigma)
mean = (up+low) / 2.0
pm = (up-low) / 2.0
print("2040:", round(mean, 3), "+/-", round(pm, 3))

# Fit the logistic fuction for df3
param, covar = opt.curve_fit(logistics, df3['Year'], df3['Urban population (% of total population)'], p0 = (2e9, 0.05, 1960.0), maxfev = 1000)
df3['fit'] = logistics(df3['Year'], *param)

sigma = np.sqrt(np.diag(covar))

# Set upper and lower limit for df3
low, up = err.err_ranges(df3["Year"], logistics, param, sigma)

# Plot the graph with original data & logistic fuction data with lower & upper limit for df3
plt.figure(dpi = 300)  # To create figure with required resolution 
plt.plot(df3['Year'], df3['Urban population (% of total population)'], label = "Ireland's urban population (%)") # To plot line graph with actual data
plt.plot(df3['Year'], df3['fit'], label = 'fit')  # To plot the 
plt.fill_between(df3['Year'], low, up, alpha = 0.7, color = 'red')
plt.xlabel('Year', fontsize = 12)
plt.ylabel("Ireland's urban population (%)", fontsize = 12)
plt.title("Ireland's urban population (%) with logistic fuction", fontsize = 15)
plt.legend()
plt.show()

# Forcast the Urban population for upcomming years with lower & upper limit for df3
print("Forcasted Ireland's urban population (%)")
low, up = err.err_ranges(2030, logistics, param, sigma)
mean = (up+low) / 2.0
pm = (up-low) / 2.0
print("2030:", round(mean, 3), "+/-", round(pm, 3))
low, up = err.err_ranges(2040, logistics, param, sigma)
mean = (up+low) / 2.0
pm = (up-low) / 2.0
print("2040:", round(mean, 3), "+/-", round(pm, 3))

# Fit the logistic fuction for Total labor force for df4
param, covar = opt.curve_fit(logistics, df4['Year'], df4['Labor force, total'], p0 = (2e9, 0.05, 1960.0), maxfev = 1000)
df4['fit'] = logistics(df4['Year'], *param)

sigma = np.sqrt(np.diag(covar))

# Set upper and lower limit for df4
low, up = err.err_ranges(df4["Year"], logistics, param, sigma)

# Plot the graph with original data & logistic fuction data with lower & upper limit for df4
plt.figure(dpi = 300)  # To create figure with required resolution 
plt.plot(df4['Year'], df4['Labor force, total'], label = "Ireland's total Labor force") # To plot line graph with actual data
plt.plot(df4['Year'], df4['fit'], label = 'fit')  # To plot the 
plt.fill_between(df4['Year'], low, up, alpha = 0.7, color = 'red')
plt.xlabel('Year', fontsize = 12)
plt.ylabel("Ireland's total Labor force", fontsize = 12)
plt.title("Ireland's total Labor force with logistic fuction", fontsize = 15)
plt.legend()
plt.show()

# Forcast the labor force for upcomming years with lower & upper limit for df4
print("Forcasted Ireland's total Labor force")
low, up = err.err_ranges(2030, logistics, param, sigma)
mean = (up+low) / 2.0
pm = (up-low) / 2.0
print("2030:", round(mean, 3), "+/-", round(pm, 3))
low, up = err.err_ranges(2040, logistics, param, sigma)
mean = (up+low) / 2.0
pm = (up-low) / 2.0
print("2040:", round(mean, 3), "+/-", round(pm, 3))

# Fit the logistic fuction for df5
param, covar = opt.curve_fit(logistics, df5['Year'], df5['Urban population (% of total population)'], p0 = (2e9, 0.05, 1960.0), maxfev = 1000)
df5['fit'] = logistics(df5['Year'], *param)

sigma = np.sqrt(np.diag(covar))

# Set upper and lower limit for df5
low, up = err.err_ranges(df5["Year"], logistics, param, sigma)

# Plot the graph with original data & logistic fuction data with lower & upper limit for df5
plt.figure(dpi = 300)  # To create figure with required resolution 
plt.plot(df5['Year'], df5['Urban population (% of total population)'], label = "Egypt's urban population (%)") # To plot line graph with actual data
plt.plot(df5['Year'], df5['fit'], label = 'fit')  # To plot the 
plt.fill_between(df5['Year'], low, up, alpha = 0.7, color = 'yellow')
plt.xlabel('Year', fontsize = 12)
plt.ylabel("Egypt's urban population (%)", fontsize = 12)
plt.title("Egypt's urban population (%) with logistic fuction", fontsize = 15)
plt.legend()
plt.show()

# Forcast the Urban population for upcomming years with lower & upper limit for df5
print("Forcasted Egypt's urban population (%)")
low, up = err.err_ranges(2030, logistics, param, sigma)
mean = (up+low) / 2.0
pm = (up-low) / 2.0
print("2030:", round(mean, 3), "+/-", round(pm, 3))
low, up = err.err_ranges(2040, logistics, param, sigma)
mean = (up+low) / 2.0
pm = (up-low) / 2.0
print("2040:", round(mean, 3), "+/-", round(pm, 3))

# Fit the logistic fuction for Total labor force for df6
param, covar = opt.curve_fit(logistics, df6['Year'], df6['Labor force, total'], p0 = (2e9, 0.05, 1960.0), maxfev = 1000)
df6['fit'] = logistics(df6['Year'], *param)

sigma = np.sqrt(np.diag(covar))

# Set upper and lower limit for df6
low, up = err.err_ranges(df6["Year"], logistics, param, sigma)

# Plot the graph with original data & logistic fuction data with lower & upper limit for df6
plt.figure(dpi = 300)  # To create figure with required resolution 
plt.plot(df6['Year'], df6['Labor force, total'], label = "Egypt's total Labor force") # To plot line graph with actual data
plt.plot(df6['Year'], df6['fit'], label = 'fit')  # To plot the 
plt.fill_between(df6['Year'], low, up, alpha = 0.7, color = 'yellow')
plt.xlabel('Year', fontsize = 12)
plt.ylabel("Egypt's total Labor force", fontsize = 12)
plt.title("Egypt's total Labor force with logistic fuction", fontsize = 15)
plt.legend()
plt.show()

# Forcast the labor force for upcomming years with lower & upper limit for df6
print("Forcasted Egypt's total Labor force")
low, up = err.err_ranges(2030, logistics, param, sigma)
mean = (up+low) / 2.0
pm = (up-low) / 2.0
print("2030:", round(mean, 3), "+/-", round(pm, 3))
low, up = err.err_ranges(2040, logistics, param, sigma)
mean = (up+low) / 2.0
pm = (up-low) / 2.0
print("2040:", round(mean, 3), "+/-", round(pm, 3))
