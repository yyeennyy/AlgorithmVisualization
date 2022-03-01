import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
from IPython.display import Image as Img
from IPython.display import display

def draw_matrix_gif(matrix_list, N, M, frame=800, show=False):

  dir_name = './temp_your_matrix'
  num_img_data = np.load('/content/MachineLearning/num_img_data.npy')

  def get_num_pix(num):
    try:
      if (num < 0 or num > 9):
        raise Exception("한 자리 수(0~9)로만 이루어진 정수 배열이어야 합니다")
    except Exception as e:
      print('예외 발생: ', e)
    return num_img_data[num]


  def makedirs():
      try:
        if not os.path.exists(dir_name):
          os.makedirs(dir_name)
        else:
          os.rmdir('file_name')
          print("existing directory " + dir_name 
                + " removed now before making dir same name")
          os.makedirs(dir_name)

      except OSError:
        return print("Error: can't makedirs " + dir_name + "\n",
            "if your existing directory '" + dir_name + "' has any files, this ERROR can be pop up")
        

  def make_GIF():
    os.chdir(dir_name)
    img_list = os.listdir()
    img_list = sorted([file_name for file_name in img_list])
    images = [Image.open(file_name) for file_name in img_list]
    im = images[0]

    os.chdir('../')
    im.save('matrixs.gif', save_all=True, append_images=images[1:], loop=0xff, duration=frame)


  def delete_tmp_files(dir, length):
    for i in range(length):
      os.remove(dir + "/{0:05d}.png".format(i))
    os.rmdir(dir)


  length = len(matrix_list)
  makedirs()
  os.chdir(dir_name)
  for count in range(length):
    this_matrix = matrix_list[count]
    fig, axs = plt.subplots(N, M)
    for i in range(N):
      for j in range(M):
        axs[i, j].imshow(get_num_pix(this_matrix[i][j]))
        axs[i, j].axis('off')
    plt.savefig("{0:05d}.png".format(count))
    if(show): plt.show()
    if(not show): plt.close(fig)
  os.chdir("../")
  make_GIF()
  delete_tmp_files(dir_name, length)

