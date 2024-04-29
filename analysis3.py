import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Створення тестових даних
np.random.seed(42)
data = pd.DataFrame({
    'Democracy Score': np.random.randint(0, 100, 100),
    'Economy Score': np.random.randint(0, 100, 100)
})

# Зменшення обсягу даних
data = data.head(20)

# Виконання кластерного аналізу
X = data.values
kmeans = KMeans(n_clusters=4)
kmeans.fit(X)

# Виведення інформації про кластери
print("Cluster Centers:")
print(kmeans.cluster_centers_)
print("\nCluster Labels:")
print(kmeans.labels_)

# Візуалізація кластерів
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='viridis', edgecolor='k', s=100)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], marker='x', s=300, c='red', label='Centroids')
plt.xlabel('Democracy Score')
plt.ylabel('Economy Score')
plt.title('Clustering of Countries')
plt.legend()
plt.grid(True)
plt.show()
