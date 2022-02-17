# parameter 'data' : numpy array OR pd.DataFrame

def kmeans_process_2d(data, n_clusters, palette = None):
  import seaborn as sns
  import scipy as sp
  import pandas as pd
  import numpy as np

  if type(data) == type(np.array([])):
    data = pd.DataFrame(data, columns=['x', 'y']).reset_index(drop=True)
  else:
    data.columns = ['x', 'y']
    data.reset_index(drop=True)

  # random centroid 
  centroids = data.sample(n_clusters).sort_values('x').reset_index(drop=True)

  # random centroid - scatter
  print("\nrandom centroid")
  plt.subplots()
  sns.scatterplot(x='x', y='y', data=data)  
  plt.scatter(centroids['x'], centroids['y'], marker='D', c='black') 
  plt.title('k-means algorithm', fontsize=15)
  plt.show()


  while(True):
    # reassign data
    distance = sp.spatial.distance.cdist(data, centroids, "euclidean")
    cluster_num = np.argmin(distance, axis=1)
    result = data.copy()
    result["cluster"] = np.array(cluster_num)

    # reassign data - scatter
    print("\nreassign data")
    plt.subplots()
    sns.scatterplot(x="x", y="y", hue="cluster", data=result, palette = palette)
    plt.scatter(centroids['x'], centroids['y'], marker='D', c='black')
    plt.title('k-means algorithm', fontsize=15)
    plt.show()
    
    # reassign centroid
    centroids_ = result.groupby("cluster").mean()
    centroids_ = pd.DataFrame(centroids_, columns=['x', 'y']).sort_values('x').reset_index(drop=True)
    if (centroids_['x'].tolist() == centroids['x'].tolist() ): break
    centroids = centroids_

    # reassign centroid - scatter
    print("\nreassign centroid")
    plt.subplots()
    sns.scatterplot(x="x", y="y", hue="cluster", data=result, palette = palette)
    plt.scatter(centroids['x'], centroids['y'], marker='D', c='black')
    plt.title('k-means algorithm', fontsize=15)
    plt.show()

  print("\ndone")
