import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# X, y = make_blobs(n_samples=100, centers=5, random_state=101)
#
#
# elbow=[]
# for i in range(1, 20):
#     kmeans = KMeans(n_clusters = i, init = 'k-means++',n_init=1, random_state = 101)
#     kmeans.fit(X)
#     elbow.append(kmeans.inertia_)
# sns.lineplot(range(1, 20), elbow,color='blue')
# plt.rcParams.update({'figure.figsize':(10,7.5), 'figure.dpi':100})
# plt.title('ELBOW METHOD')
# plt.show()


df=pd.read_csv("Mall_Customers.csv")
df.head()
X = df.iloc[:, [2, 4]].values
from sklearn.cluster import KMeans
elbow=[]
for i in range(1, 20):
    kmeans = KMeans(n_clusters = i, init = 'k-means++',n_init=1, random_state = 101)
    kmeans.fit(X)
    elbow.append(kmeans.inertia_)


print(elbow)
print(range(1, 20))
sns.lineplot(elbow)
plt.rcParams.update({'figure.figsize':(10,7.5), 'figure.dpi':100})
plt.title('ELBOW METHOD')
plt.show()


kmeans = KMeans(n_clusters = 5, init = 'k-means++',n_init=1, random_state = 101)
y_pred = kmeans.fit_predict(X)
plt.figure(figsize=(15,7.5))
sns.scatterplot(x=X[y_pred == 0, 0], y=X[y_pred == 0, 1],s=50)
sns.scatterplot(x=X[y_pred == 1, 0], y=X[y_pred == 1, 1],s=50)
sns.scatterplot(x=X[y_pred == 2, 0], y=X[y_pred == 2, 1],s=50)
sns.scatterplot(x=X[y_pred == 3, 0], y=X[y_pred == 3, 1],s=50)
sns.scatterplot(x=X[y_pred == 4, 0], y=X[y_pred == 4, 1],s=50)
plt.title('Clusters')
plt.legend()

plt.show()
sns.scatterplot(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],s=500,color='yellow')



