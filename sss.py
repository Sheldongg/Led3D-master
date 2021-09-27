import cv2
import numpy as np
import matplotlib.pyplot as plt
img_path="/home/alien/Downloads/Azure-Kinect-Samples/muild/bin/depth/lidong0.5/10.png"
uint8_img = cv2.imread(img_path)
# plt.imshow(uint8_img,"gray",vmin=0,vmax=4096)
# plt.show()
#cv2.imshow('1', uint8_img,vmin=0,vmax=4096)
img_read = cv2.imread(img_path, 1)
img_norm = np.zeros(img_read.shape)

x=cv2.normalize(img_read, dst=img_norm, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
cv2.imshow('ssss',x)
uint16_img = cv2.imread(img_path, -1)
# cv2.imshow('2', uint16_img)
uint16_img -= uint16_img.min()
# cv2.imshow('3', uint16_img)

uint16_img = uint16_img / (uint16_img.max() - uint16_img.min())
cv2.imshow('4', uint16_img)
uint16_img *= 255
cv2.imshow('5', uint16_img)
new_uint16_img = uint16_img.astype(np.uint8)
cv2.imshow('UINT8', uint8_img)
cv2.imshow('UINT16', new_uint16_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imshow('UINT16', new_uint16_img)