import numpy as np
import cv2
def np_list_int(tb):
    tb_2 = tb.tolist()  # 将np转换为列表
    return tb_2

def shot(img, dt_boxes):  # 应用于predict_det.py中,通过dt_boxes中获得的四个坐标点,裁剪出图像
    dt_boxes = np_list_int(dt_boxes)
    boxes_len = len(dt_boxes)
    num = 0
    while 1:
        if (num < boxes_len):
            box = dt_boxes[num]
            tl = box[0]
            tr = box[1]
            br = box[2]
            bl = box[3]
            print("打印转换成功数据num =" + str(num))
            print("tl:" + str(tl), "tr:" + str(tr), "br:" + str(br), "bl:" + str(bl))
            print(tr[1], bl[1], tl[0], br[0])

            crop = img[int(tr[1]):int(bl[1]), int(tl[0]):int(br[0])]

            # crop = img[27:45, 67:119] #测试
            # crop = img[380:395, 368:119]

            cv2.imwrite("K:/paddleOCR/PaddleOCR/screenshot/a/" + str(num) + ".jpg", crop)

            num = num + 1
        else:
            break
def shot1(img_path, tl, tr, br, bl, i):
    tl = np_list_int(tl)
    tr = np_list_int(tr)
    br = np_list_int(br)
    bl = np_list_int(bl)

    print("打印转换成功数据")
    print("tl:" + str(tl), "tr:" + str(tr), "br:" + str(br), "bl:" + str(bl))

    img = cv2.imread(img_path)
    crop = img[tr[1]:bl[1], tl[0]:br[0]]

    # crop = img[27:45, 67:119]

    cv2.imwrite("/home/alien/Downloads/sd/" + str(i) + ".jpg", crop)

# tl1 = np.array([67,27])
# tl2= np.array([119,27])
# tl3 = np.array([119,45])
# tl4 = np.array([67,45])
# shot("K:\paddleOCR\PaddleOCR\screenshot\zong.jpg",tl1, tl2 ,tl3 , tl4 , 0)
if __name__ == '__main__':
    img_path="/home/alien/Downloads/Azure-Kinect-Samples/muild/bin/depth/lidong0.5/10.png"
    #img_path="/home/alien/Downloads/v2-e544adfc65836e4f737aa3e6a9c175ea_720w.jpg"
    img=cv2.imread(img_path,3)
    cv2.imshow("ex",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    tr=np.array([267,138])
    bl=np.array([447,318])
    tl=np.array([261,318])
    br=np.array([447,138])
    i=0
    shot1(img_path,tl,tr,br,bl,i)

