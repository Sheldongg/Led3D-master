import sys
sys.path.append("NumPy_path")
import cv2
import scipy.misc
import numpy as np
import pcl
import math
import pcl.pcl_visualization
import matplotlib.image as mpimg
# from imresize import *
from PIL import Image

# import pcl.pcl_visualization
# p = pcl.PointCloud(10)  # "empty" point cloud
# a = np.asarray(p)       # NumPy view on the cloud
# a[:] = 0                # fill with zeros
# print(p[3])             # prints (0.0, 0.0, 0.0)
# a[:, 0] = 1             # set x coordinates to 1
# print(p[3])             # prints (1.0, 0.0, 0.0)

depth = cv2.imread("/home/alien/Downloads/Azure-Kinect-Samples/muild/bin/depth/lidong0.5/10.png",-1)
# depth=imread('/home/alien/Downloads/Azure-Kinect-Samples/muild/bin/depth/lidong0.5/10.png')
cloud = pcl.PointCloud()
rows = len(depth)
cols = len(depth[0])
pointcloud=[]
#x:357 y:228
crop = depth[138:318, 267:447]#true
# ceo=np.array(crop)
# cv2.resizeWindow("",180,180)
cv2.imshow("s",crop)

# cv2.destroyAllWindows()
# depth=cv2.split(crop)[0]
# depth[depth>2000]=0
# depth=depth/2000.0000
# cv2.imshow('imgOri',depth)
# cv2.waitKey(0)
rreSize=360
dst=scipy.misc.imresize(crop.astype(np.uint16),(rreSize,rreSize),mode='F').astype(np.uint16)
ceo=np.array(dst)
# dst=crop.resize([360,360])
# dst=imresize(crop,[360,360])
# im = np.array(Image.fromarray(im).resize((h, int(w * aspect_ratio))))
pc_template=np.zeros((3,rreSize*rreSize))
a=np.zeros(129600)
b=np.zeros(129600)
c=np.zeros(129600)
print("des")
for i in range(129600):
    a[i]=math.floor(i/360)+1
for i in range(129600):
    b[i]=i%360+1


pc_template[0,:] =a
pc_template[1,:]=b
#
# for i in range(129600):
#      for j in range(360):
#          for m in range(360):
#              pc_template[2,i]=dst[m][j]
             # print(pc_template[2,i])

pc_template[2,:]=ceo.reshape(1,-1)
# p = pcl.PointCloud(pc_template[:])
print("dess")
# pc_template(2,:) = mod(0: (reSize * reSize - 1), reSize)+1;
# pc_template(3,:) = roi_face(:)
# for m in range(0,rows):
#     for n in range(0,cols):
#         d = depth[m][n][0] + depth[m][n][1] * 256
#         if d == 0:
#             pass
#         else:
#             z = float(d)
#             x = n * z / camera_fx
#             y = m * z / camera_fy
#             points = [x, y, z]
#             pointcloud.append(points)
#
#     # 由于pcl库不会直接识别列表格式的点云数据，所以我们需要使用numpy库进行数据格式转换，并将点云保存pcd文件中。
#
#
pointcloud = np.array(pc_template, dtype=np.float32)
cloud.from_array(pointcloud)
pcl.save(cloud, "cloud.pcd", format='pcd')
pcl.load_PointWithViewpoint("./cloud.pcd", format=None)