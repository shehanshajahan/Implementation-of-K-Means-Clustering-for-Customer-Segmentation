# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: Shehan Shajahan
RegisterNumber: 212223240154
*/
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
#Load data from CSV
data = pd.read_csv("/content/Mall_Customers_EX8.csv")
data
#Extract features
X = data[['Annual Income (k$)','Spending Score (1-100)']]
X
plt.figure(figsize=(4,4))
plt.scatter(data['Annual Income (k$)'],data['Spending Score (1-100)'])
plt.xlabel('Annual Income(k$)')
plt.ylabel('Spending Score (1-100)')
plt.show()
#Number of clusters
k = 5
#Initialize KMeans
kmeans = KMeans(n_clusters=k)
#Fit the data
kmeans.fit(X)
centroids = kmeans.cluster_centers_
#Get the cluster labels for each data point
labels = kmeans.labels_
print("Centroids:")
print(centroids)
print("Labels:")
print(labels)
colors = ['r','g','b','c','m'] #Define colors for each cluster
for i in range(k):
  cluster_points=X[labels==i] #Get data points belonging to cluster i
  plt.scatter(cluster_points['Annual Income (k$)'],cluster_points['Spending Score (1-100)'],
              color=colors[i],label=f'Cluster(i+1)')
  #Find minimum enclosing circle
distances=euclidean_distances(cluster_points,[centroids[i]])
radius=np.max(distances)
circle=plt.Circle(centroids[i],radius,color=colors[i],fill=False)
plt.gca().add_patch(circle)
#Plotting the centroids
plt.scatter(centroids[:,0],centroids[:,1],marker='*',s=200,color='k',label='Centroids')
plt.title('K-means Clustering')
plt.xlabel('Annual Income (k$)')
plt.legend()
plt.grid(True)
plt.axis('equal') #Ensure aspect ratio is equal
plt.show()
```

## Output:
## DATASET:
![Screenshot 2024-04-16 105303](https://github.com/shehanshajahan/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/139317389/d1d27e0e-e107-42fd-aaa6-0fb847cf24dd)
## CENTROIDS AND LABELS:
![Screenshot 2024-04-16 105240](https://github.com/shehanshajahan/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/139317389/986d0fdf-a82e-4265-a377-991b8c87e4d8)
## GRAPH:
![Screenshot 2024-04-16 105249](https://github.com/shehanshajahan/Implementation-of-K-Means-Clustering-for-Customer-Segmentation/assets/139317389/213ce817-ab83-4a4c-8bcb-075d0cd95f37)


## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
