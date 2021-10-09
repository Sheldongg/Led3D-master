# import sys
# sys.path.append("NumPy_path")
import cv2
from scipy import misc
import numpy as np
import math
import dep
import pcl
import os
import linecache
from re import findall
import inference as inf
import mxnet as mx
import glob
from mxnet import gluon
from sklearn.metrics.pairwise import cosine_similarity
# 传入点云对象
# def points2pcd(points):
#     # 存放路径
#     PCD_DIR_PATH = os.path.join(os.path.abspath('.'), 'pcd')
#     # print(PCD_DIR_PATH)
#     PCD_FILE_PATH = os.path.join(PCD_DIR_PATH, 'nnache.pcd')
#     # print(PCD_FILE_PATH)
#     if os.path.exists(PCD_FILE_PATH):
#         os.remove(PCD_FILE_PATH)
#     # 写文件句柄
#     handle = open(PCD_FILE_PATH, 'a')
#
#     # 得到点云点数
#     point_num = points.shape[0]
#
#     # pcd头部（重要）
#     handle.write(
#         '# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1')
#     string = '\nWIDTH ' + str(point_num)
#     handle.write(string)
#     handle.write('\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0')
#     string = '\nPOINTS ' + str(point_num)
#     handle.write(string)
#     handle.write('\nDATA ascii')
#
#     # 依次写入点
#     for i in range(point_num):
#         string = '\n' + str(points[i, 0]) + ' ' + str(points[i, 1]) + ' ' + str(points[i, 2])
#         handle.write(string)
#     handle.close()


if __name__ == "__main__":
     # for i in range(1,2):
        x=10
        str1="/home/alien/Downloads/Azure-Kinect-Samples/muild/bin/depth/lidong0.5/"+str(x)+".png"
        str2 = "/home/alien/Downloads/Azure-Kinect-Samples/muild/bin/lidong0.5/" + str(x) + ".txt"

        depth = cv2.imread(str1,-1)
        thline=linecache.getline(str2,3)

        pattern=r'\d+'
        parass=findall(pattern,thline)
        x=int(parass[0])
        y=int(parass[1])
        # depth=imread('/home/alien/Downloads/Azure-Kinect-Samples/muild/bin/depth/lidong0.5/10.png')
        cloud = pcl.PointCloud()
        rows = len(depth)
        cols = len(depth[0])
        pointcloud=[]
        #x:357 y:228
        crop = depth[y-90:y+90, x-90:x+90]#true

        rreSize=360
        dst = cv2.resize(crop, [rreSize, rreSize],interpolation=cv2.INTER_LINEAR)
        ceo=np.array(dst)
        beo=ceo.T
        beo=beo.reshape(-1,1)
        beo=beo.T

        pc_template=np.zeros((3,rreSize*rreSize))
        a=np.zeros(129600)
        b=np.zeros(129600)
        c=np.zeros(129600)
        # print("des")
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
        # points2pcd(mmmm)
        r = 100
        xo = int(rreSize / 2)
        yo = int(rreSize / 2)
        m=np.median(dst[xo - 10:xo + 10, yo - 10: yo + 10])
        mi=dst[xo - 10:xo + 10, yo - 10: yo + 10]
        zo = np.median(np.median(mi,axis=0))
        # zo = 634;
        pc_template[2, ((xo - pc_template[1,:])* (xo - pc_template[1,:]) + (yo - pc_template[0,:])*(yo - pc_template[0,:])+(zo - pc_template[2,:])*(zo - pc_template[2,:]))  > r * r]=0
        pc_face = pc_template[:, pc_template[2,:] > 0]
        pc_face=np.array(pc_face)
        nnnn=pc_face.T
        # points2pcd(nnnn)#gllery point face

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
        misc.imsave('/home/alien/sol/2.jpg',finshed_pic)
        # cv2.imshow("ssd", im_rotate)
        # cv2.waitKey(0)

        '''calculate similarity'''
        ctx = mx.cpu(0)  # all cpu is computing in default and 0 is machine ID
        mod = inf.load_mod(ctx)
        # n = gluon.data.vision.ImageFolderDataset('/home/alien/Downloads/lidong/Led3D-master/python/data/1gallery',
        #                                          flag=1, transform=None)
        # print(len(n))
        # print(n[0])
        evalue = []
        # n is the total of recognition
        # m is the total of correct
        m = 0
        n = 0
        # for filenames in glob.glob(r'./data/0.5probe/*/*'):
        for filenames in glob.glob(r'/home/alien/sol/2.jpg'):
            text = {}
            pattern1 = r'\d+.jpg'
            paras = findall(pattern1, filenames)
            # print("use constrast is", paras[0])
            p = inf.extract_feature(filenames,mod)
            for filename in glob.glob(r'/home/alien/Downloads/lidong/Led3D-master/python/data/0.5gallery/*/*.jpg'):
                pattern = r'/\d+/'
                parass = findall(pattern, filename)
                # print(parass)
                g = inf.extract_feature(filename,mod)
                sim = cosine_similarity([g], [p])
                text[parass[0]] = sim[0][0]
                # print(filename + ":" + str(sim[0][0]))

            z = list(text.keys())[list(text.values()).index(max(text.values()))]
            print(z)
