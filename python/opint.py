# 我们先随机生成一些数字,作为点云输入,为了减少物体尺度的问题,
#通常会将点云缩到半径为1的球体中
#为了方便起见,LZ把batch_size改成1
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import  Axes3D
point_cloud = np.random.rand(1024, 3) - np.random.rand( 1024, 3)

input_images = np.zeros((500, 500))
filename = "/home/alien/Downloads/Azure-Kinect-Samples/muild/bin/depth/lidong0.5/10.png"
f=open(filename)

#画出3d点云
def pyplot_draw_point_cloud(points, output_filename=None):
    """ points is a Nx3 numpy array """
    auto_add_to_figure = False
    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')
    # ax = Axes3D(fig,auto_add_to_figure=False)
    fig.add_axes(ax)
    ax.scatter(points[:, 0], points[:, 1], points[:, 2])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()
    # savefig(output_filename)
if __name__ == "__main__":
   pyplot_draw_point_cloud(point_cloud)