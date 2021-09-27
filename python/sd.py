import numpy as np
import cv2
from PIL import Image
import scipy.signal as signal
import matplotlib.pyplot as plt

# 创建一个500*500的矩阵
input_images = np.zeros((500, 500))
filename = "/home/alien/Downloads/Azure-Kinect-Samples/muild/bin/depth/lidong0.5/10.png"
# convert将当前图像转换为灰度模式，并且返回新的图像。
# 将图片在重新定义的矩阵中再显示，不然可能会只显示部分。
img = Image.open(filename).resize((500, 500)).convert('L')
plt.subplot(221)
# plt.title('原图', fontproperties=font_set)
plt.imshow(img)



# 图像的尺寸，按照像素数计算。它的返回值为宽度和高度的二元组（width, height）。
width = img.size[0]
height = img.size[1]
threshold = 130
# 可以改写代码使其成为二值化,此代码可理解为反向二值化
for h in range(height):
    for w in range(width):
        # getpixel直接获得（h，w）处的像素直接返回这个点三个通道的像素值
        # 返回给定位置的像素值。如果图像为多通道，则返回一个元组(r,g,b,阈值）。
        # 如果改成（w，h）出现的图像会倒转
        if img.getpixel((w, h)) < threshold:

            input_images[h, w] = 1
        else:
            input_images[h, w] = 0
plt.subplot(222)
# plt.title('二值化', fontproperties=font_set)
plt.imshow(input_images)

data = signal.medfilt2d(np.array(img), kernel_size=3)  # 二维中值滤波
for h in range(0, height):
    for w in range(0, width):
        if data[h][w] < 128:
            input_images[h, w] = 0
        else:
            input_images[h, w] = 1

plt.subplot(223)
# plt.title('中值滤波去噪（3*3）', fontproperties=font_set)
plt.imshow(input_images)

data = signal.medfilt2d(np.array(img), kernel_size=7)  # 二维中值滤波
for h in range(0, height):
    for w in range(0, width):
        if data[h][w] < 128:
            input_images[h, w] = 0
        else:
            input_images[h, w] = 1
plt.subplot(224)
# plt.title('中值滤波去噪（7*7）', fontproperties=font_set)
plt.imshow(input_images)
plt.show()
