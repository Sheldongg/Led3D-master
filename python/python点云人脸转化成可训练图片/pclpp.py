import sys
sys.path.append("NumPy_path")
import cv2
from scipy import  misc
import numpy as np
import math
import dep
import pcl
import os
# 传入点云对象
def points2pcd(points):
    # 存放路径
    PCD_DIR_PATH = os.path.join(os.path.abspath('.'), 'pcd')
    print(PCD_DIR_PATH)
    PCD_FILE_PATH = os.path.join(PCD_DIR_PATH, 'nnache.pcd')
    print(PCD_FILE_PATH)
    if os.path.exists(PCD_FILE_PATH):
        os.remove(PCD_FILE_PATH)
    # 写文件句柄
    handle = open(PCD_FILE_PATH, 'a')

    # 得到点云点数
    point_num = points.shape[0]

    # pcd头部（重要）
    handle.write(
        '# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1')
    string = '\nWIDTH ' + str(point_num)
    handle.write(string)
    handle.write('\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0')
    string = '\nPOINTS ' + str(point_num)
    handle.write(string)
    handle.write('\nDATA ascii')

    # 依次写入点
    for i in range(point_num):
        string = '\n' + str(points[i, 0]) + ' ' + str(points[i, 1]) + ' ' + str(points[i, 2])
        handle.write(string)
    handle.close()


if __name__ == "__main__":
    depth = cv2.imread("/home/alien/Downloads/Azure-Kinect-Samples/muild/bin/depth/lidong0.5/10.png",-1)
    # depth=imread('/home/alien/Downloads/Azure-Kinect-Samples/muild/bin/depth/lidong0.5/10.png')
    cloud = pcl.PointCloud()
    rows = len(depth)
    cols = len(depth[0])
    pointcloud=[]
    #x:357 y:228
    crop = depth[138:318, 267:447]#true

    rreSize=360
    dst = cv2.resize(crop, [rreSize, rreSize],interpolation=cv2.INTER_LINEAR)
    ceo=np.array(dst)
    beo=ceo.T
    beo=beo.reshape(-1,1)
    beo=beo.T
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
    pc_template[2,:]=beo
    pc_template=np.array(pc_template)
    mmmm=pc_template.T
    # teo=np.array(pc_template)    # # teo=teo.T
    points2pcd(mmmm)
    r = 100
    xo = int(rreSize / 2)
    yo = int(rreSize / 2)
    m=np.median(dst[xo - 10:xo + 10, yo - 10: yo + 10])
    mi=dst[xo - 10:xo + 10, yo - 10: yo + 10]
    zo = np.median(np.median(mi,axis=0));
    # zo = 634;
    pc_template[2, ((xo - pc_template[1,:])* (xo - pc_template[1,:]) + (yo - pc_template[0,:])*(yo - pc_template[0,:])+(zo - pc_template[2,:])*(zo - pc_template[2,:]))  > r * r]=0
    pc_face = pc_template[:, pc_template[2,:] > 0]
    pc_face=np.array(pc_face)
    nnnn=pc_face.T
    points2pcd(nnnn)#gllery point face

    '''calcDepthAndNormal'''
    nnnn[:,2]=max(nnnn[:,2])-nnnn[:,2]
    sd=max(nnnn[:, 0]) - min(nnnn[:, 0])
    sb=max(nnnn[:,1])-min(nnnn[:,1])
    sd = np.array(sd, dtype='uint8')
    sb = np.array(sb, dtype='uint8')
    depth=dep.rret(nnnn,sd,sb)
    mask=np.array(depth)
    nn=mask.shape[0]
    mm=mask.shape[1]
    max_x=nn
    min_y=0
    min_x=0
    max_y=mm
    croped_face=depth[min_x:max_x,min_y:max_y]
    depth=croped_face
    croped_mask=mask[min_x:max_x,min_y:max_y]
    data = np.array(depth, dtype='uint8')
    mask=croped_mask
    im_rotate = misc.imrotate(data,-90)
    reSize=180
    finshed_pic = cv2.resize(im_rotate, [reSize, reSize], interpolation=cv2.INTER_LINEAR)
    misc.imsave('/home/alien/sd.jpg',finshed_pic)
    cv2.imshow("ssd", im_rotate)
    cv2.waitKey(0)

