import seaborn as sns
import scipy as sp
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



# parameter 'data' : numpy array OR pd.DataFrame OR python list
def kmeans_process_2d(data, n_clusters, palette = None):

  if type(data) == type(pd.DataFrame([])):
    data.columns = ['x', 'y']
    data.reset_index(drop=True)
  else: 
    data = pd.DataFrame(data).reset_index(drop=True)
    data.columns = ['x', 'y']
    
  # random centroid 
  centroids = data.sample(n_clusters).sort_values('x').reset_index(drop=True)

  # random centroid - scatter
  print("\nrandom centroid")
  plt.subplots()
  sns.scatterplot(x='x', y='y', data=data)  
  plt.scatter(centroids['x'], centroids['y'], marker='D', c='black') 
  plt.title('k-means algorithm', fontsize=15)
  plt.show()

  def drawFigure():
    fig = plt.figure()
    sns.scatterplot(x="x", y="y", hue="cluster", data=result, palette = palette,  
                    legend = False)
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
    drawFigure()
    
    # reassign centroid
    centroids_ = result.groupby("cluster").mean()
    centroids_ = pd.DataFrame(centroids_, columns=['x', 'y']).sort_values('x').reset_index(drop=True)
    if (centroids_['x'].tolist() == centroids['x'].tolist() and
        centroids_['y'].tolist() == centroids['y'].tolist()): break
    centroids = centroids_

    # reassign centroid - scatter
    print("\nreassign centroid")
    drawFigure()

  print("\ndone")


 

def gif_kmeans(data, n_clusters, palette, frame=1000):
  import os
  from PIL import Image
  from IPython.display import Image as Img
  from IPython.display import display


  if type(data) == type(pd.DataFrame([])):
    data.columns = ['x', 'y']
    data.reset_index(drop=True)
  else: 
    data = pd.DataFrame(data).reset_index(drop=True)
    data.columns = ['x', 'y']

  count = 0

  # makedirs
  try:
    if not os.path.exists('kmeans'):
      os.makedirs('kmeans')
    else:
      os.rmdir('./kmeans')
      print("existing directory 'kmeans' removed now before making dir 'kmeans'")
      os.makedirs('kmeans')
  except OSError:
    return print("Error: can't makedirs ./kmeans \n",
          "if your existing directory 'kmeans' has any files, this ERROR can be pop up")


  # change working dir: ./kmeans
  os.chdir('./kmeans')

  # random centroid 
  centroids = data.sample(n_clusters).sort_values('x').reset_index(drop=True)

  # random centroid - scatter
  fig = plt.figure()
  sns.scatterplot(x='x', y='y', data=data)
  plt.scatter(centroids['x'], centroids['y'], marker='D', c='black') 
  plt.title('k-means algorithm', fontsize=15)
  plt.savefig("{0:05d}.png".format(count))
  plt.close(fig)
  count += 1
  
  def drawFigure():
    fig = plt.figure()
    sns.scatterplot(x="x", y="y", hue="cluster", data=result, palette = palette,  
                    legend = False)
    plt.scatter(centroids['x'], centroids['y'], marker='D', c='black')
    plt.title('k-means algorithm', fontsize=15)
    return fig

  while(True):
    # reassign data
    distance = sp.spatial.distance.cdist(data, centroids, "euclidean")
    cluster_num = np.argmin(distance, axis=1)
    result = data.copy()
    result["cluster"] = np.array(cluster_num)

    # reassign data - scatter
    fig = drawFigure()
    plt.savefig("{0:05d}.png".format(count))
    plt.close(fig)
    count += 1
    
    # reassign centroid
    centroids_ = result.groupby("cluster").mean()
    centroids_ = pd.DataFrame(centroids_, columns=['x', 'y']).sort_values('x').reset_index(drop=True)
    if (centroids_['x'].tolist() == centroids['x'].tolist() and
        centroids_['y'].tolist() == centroids['y'].tolist()): break
    centroids = centroids_

    # reassign centroid - scatter
    fig = drawFigure()
    plt.savefig("{0:05d}.png".format(count))
    print((int) (count/2))
    plt.close(fig)
    count += 1


  # start to make gif file
  # os.listdir() : return all files or dirs in working directory
  img_list = os.listdir()
  img_list = sorted([file_name for file_name in img_list])
  images = [Image.open(file_name_with_path) for file_name_with_path in img_list]
  im = images[0]

  # return root dir
  os.chdir('../')
  im.save('k-means.gif', save_all=True, append_images=images[1:], loop=0xff, duration=frame)  # duration: 프레임 전환 속도

  # delete tmp files
  for i in range(count):
    os.remove("./kmeans/{0:05d}.png".format(i))
  os.rmdir('./kmeans')
  
  return print('done : save success')
