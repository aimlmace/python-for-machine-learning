import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


data = pd.read_csv('/home/cs-ai-25/exp-ml/Mall_Customers.csv')

X = data.iloc[:,[3,4]].values

kmeans = KMeans(n_clusters=5,random_state=25)

kmeans.fit(X)

pred = kmeans.predict(X)

print(pred)

data['cluster'] = pred

income = input('Enter Annual Income: ')
score = input('Enter Spending Score: ')

user_input = {
    'Annual Income (k$)': [float(income)],
    'Spending Score (1-100)':[float(score)]
}

user_input_df = pd.DataFrame(user_input)

user_pred = kmeans.predict(user_input_df.values)

print(f'The Customer Cluster is : {user_pred[0]}')
data.to_csv('/home/cs-ai-25/exp-ml/Mall_Customers_with_clusters.csv', index=False)

plt.figure(figsize=(10, 6))


plt.scatter(X[pred == 0, 0], X[pred == 0, 1], s=50, c='red', label='Cluster 1')
plt.scatter(X[pred == 1, 0], X[pred == 1, 1], s=50, c='blue', label='Cluster 2')
plt.scatter(X[pred == 2, 0], X[pred == 2, 1], s=50, c='green', label='Cluster 3')
plt.scatter(X[pred == 3, 0], X[pred == 3, 1], s=50, c='cyan', label='Cluster 4')
plt.scatter(X[pred == 4, 0], X[pred == 4, 1], s=50, c='magenta', label='Cluster 5')


centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1], s=100, c='black', label='Centroids', marker='X')


plt.title('Customer Clusters')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.grid()
plt.show()
