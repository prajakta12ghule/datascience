#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 14:43:53 2023

@author: myyntiimac
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
df = pd.read_csv("/Users/myyntiimac/Desktop/Mall_Customers.csv")
X=df.iloc[:,[3,4]].values
# Using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans 
#we are going to findout the optimal number of cluster for this we have to use the elbow method
wcss = [] 
#to plot the elbow metod we have to compute WCSS for 10 different number of cluster since we gonna have 10 iteration
#we are going to write a for loop to create a list of 10 different wcss for the10 number of clusters 
#thats why we have to initialise wcss[] & we start our loop 

for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
# Training the K-Means model on the dataset
kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X)
y_kmeans 
# Visualizing the clusters
colors = ['red', 'blue', 'green','black', 'purple']

for cluster_label in range(5):
    plt.scatter(X[y_kmeans == cluster_label, 0], X[y_kmeans == cluster_label, 1], s=100, c=colors[cluster_label], label=f'Cluster {cluster_label+1}')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='yellow', label='Centroids')

plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

# Step 4: Insert the predicted labels
df['Cluster'] = kmeans.labels_
# Step 5: Save the modified dataset
df.to_csv('modified_dataset.csv', index=False)  # Replace 'modified_dataset.csv' with the desired filename