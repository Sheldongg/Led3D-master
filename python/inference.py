# -*- coding:utf-8 -*-
import cv2
import mxnet as mx
import numpy as np
from collections import namedtuple
from sklearn.metrics.pairwise import cosine_similarity
import glob
from mxnet import gluon
from re import findall

Batch = namedtuple('Batch', ['data'])
def get_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    img = np.swapaxes(img, 0, 2)  #exchange two axis
    img = np.swapaxes(img, 1, 2)
    img = img[np.newaxis, :]      #add one dimendsion
    return img
def load_mod(ctx):
    sym, arg_params, aux_params = mx.model.load_checkpoint('./param/led3d', 0) #
    internals = sym.get_internals()
    fc1 = internals['s5_global_conv_output']
    group = mx.symbol.Group([fc1, sym])
    mod = mx.mod.Module(symbol=group, context=ctx,label_names=None)


    mod.bind(for_training=False, data_shapes=[('data', (1, 3, 128, 128))])
    mod.set_params(arg_params, aux_params, allow_missing=True)
    return mod
def extract_feature(path,mod):
    image = get_image(path)
    mod.forward(Batch([mx.nd.array(image)]))
    mod_out = mod.get_outputs()[0].asnumpy()
    feature = np.squeeze(mod_out)
    return feature
if __name__ == '__main__':
#def simm(pathl):
    ctx= mx.cpu(0)  #all cpu is computing in default and 0 is machine ID
    mod=load_mod()
    n=gluon.data.vision.ImageFolderDataset('/home/alien/Downloads/lidong/Led3D-master/python/data/gallery',flag=1,transform=None)
    print(len(n))
    print(n[0])
    evalue = []
    #n is the total of recognition
    #m is the total of correct
    m=0
    n=0
    for filenames in glob.glob(r'./data/0.5probe/*/*'):
        text={}
        pattern1=r'/\d+/'
        paras= findall(pattern1,filenames)
        print("use constrast is",paras)
        p = extract_feature(filenames)
        for filename in glob.glob(r'/home/alien/Downloads/lidong/Led3D-master/python/data/0.5gallery/*/*.jpg'):
            pattern = r'/\d+/'
            parass = findall(pattern, filename)
          #print(parass)

            g = extract_feature(filename)
            sim = cosine_similarity([g], [p])
            text[parass[0]]=sim[0][0]

            #print(sim[0][0])
        # for k, v in text.items():
        #     if v == max(text.values):
        #         ru=k

        z = list(text.keys())[list(text.values()).index(max(text.values()))]
        n=n+1
        if z==paras[0]:
            m=m+1
        else:
            print(z)

    print('%.3f' %(m/n))


    #     evalue.append(sim)
    # for i in evalue:
    #     print(i)