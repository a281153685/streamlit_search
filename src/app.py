# -*- coding: utf-8 -*-
# @Author  : Mumu
# @Time    : 2021/11/24 15:11
'''
@Function:
 
'''
import base64
import cv2
import numpy as np
import pandas as pd
import pymysql
import streamlit as st
import torch
from milvus import Milvus
from torch import nn
from torchvision import transforms
from torchvision.models import resnet50

IMAGE_SIZE = (224, 224)
activate = {}
# torch.cuda.set_device(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


# crop the white border
def crop_back(gray):
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)

    # 索比尔算子来计算x、y方向梯度
    gradX = cv2.Sobel(blurred, ddepth=cv2.CV_32F, dx=1, dy=0)
    gradY = cv2.Sobel(blurred, ddepth=cv2.CV_32F, dx=0, dy=1)

    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)

    blurred = cv2.GaussianBlur(gradient, (9, 9), 0)
    (_, thresh) = cv2.threshold(blurred, 40, 255, cv2.THRESH_BINARY)

    closed = cv2.dilate(thresh, None, iterations=2)

    height, width = closed.shape[:2]
    top = 99999
    bottom = 0
    left = 99999
    right = 0
    for h in range(height):
        if np.sum(closed[h, :]) > 0:
            if top > h:
                top = h
            if bottom < h:
                bottom = h

    for w in range(width):
        if np.sum(closed[:, w]) > 0:
            if left > w:
                left = w
            if right < w:
                right = w
    box = (top, bottom, left, right)
    crop_img = gray[box[0]:box[1], box[2]:box[3]]
    return crop_img


# make the image to gray
def to_gray(img):
    img = cv2.resize(img, IMAGE_SIZE)
    try:
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    except:
        img = img
    img2 = np.zeros_like(img)
    img2[:, :, 0] = grey
    img2[:, :, 1] = grey
    img2[:, :, 2] = grey
    return img2


# get activation and mean the pool
def get_activation(name):
    def hook(models, input, output):
        output1 = nn.AdaptiveAvgPool2d((1, 1))(output.detach())
        vector = torch.flatten(output1, 1, 3).cpu().numpy().tolist()
        activate[name] = vector

    return hook


def selectMysqlreg(idlist):
    imageHost = {
        'ip': '数据库ip',
        'user': '',
        'psw': '',
        'database': ''
    }
    Client = pymysql.connect(host=imageHost['ip'], user=imageHost['user'], password=imageHost['psw'],
                             database=imageHost['database'])
    num = []
    for _ in idlist:
        sql = ''' select REGNO from add_intcls where ID = {} '''.format(_)
        num.append(pd.read_sql(sql, Client)['REGNO'].values.tolist()[0])

    return num

# milvus make
class milvus_init():
    def __init__(self):
        self.client = Milvus(host='milvusip', port='19530')

    def insert_vector(self, collention, vector, indexs):
        info = self.client.insert(collection_name=collention, records=vector, ids=indexs)
        return info

    def searchVector(self, collection, vector, search_param=None, top_k=5, tag=None):
        if search_param:
            if tag:
                result = self.client.search(collection_name=collection, query_records=vector, partition_tags=tag,
                                            top_k=top_k, params=search_param)
            else:
                result = self.client.search(collection_name=collection, query_records=vector, top_k=top_k,
                                            params=search_param)
        else:
            if tag:
                result = self.client.search(collection_name=collection, query_records=vector, partition_tags=tag,
                                            top_k=top_k)
            else:
                result = self.client.search(collection_name=collection, query_records=vector, top_k=top_k)   # 耗时很大

        return result

# image to vector
class image_to_vector:
    def __init__(self):
        # self.model = torch.load(dir_save).to(device)
        self.model = resnet50(pretrained=True).to(device)
        self.model.eval()
        self.model.layer4[1].conv2.register_forward_hook(get_activation('vector'))

    def to_vector(self, img):
        self.model(img)
        vector = activate['vector']
        return vector


#   初始化操作
collections = 'gray_crop_torch'
milvus = milvus_init()
mains = image_to_vector()
stop = 0.70   # 相似度图像阈值

def img_search(img):
    crop_img = crop_back(img)
    try:
        img_new = to_gray(crop_img)
    except:
        img_new = to_gray(img)

    img = transform(img_new).to(device)
    img_torch = torch.unsqueeze(img, 0)

    vectors = mains.to_vector(img_torch)
    result = milvus.searchVector(collections, vectors, top_k=100)
    result1 = result[1]
    reg_id = result1.id_array[0]
    reg_distance = result1.distance_array[0]
    res = []

    reg = selectMysqlreg(reg_id)
    for index, i in enumerate(reg):
        if reg_distance[index] > stop:
            break
        ls = [reg[index], reg_distance[index]]
        res.extend([ls])
    print('一共找到了{}条数据'.format(len(reg)))
    return res


def main():
    st.title("展示图片相似度检索效果!")
    st.sidebar.title("请在这上传你想要检索的图片")
    st.sidebar.write(
        ""
    )
    uploaded_file = st.sidebar.file_uploader("tips:", type="jpg")
    # st.sidebar.caption(f"Streamlit version `{st.__version__}`")


    # 文件上传控件
    # 上传图片并展示
    # uploaded_file = st.file_uploader("上传一张图片", type="jpg")

    if uploaded_file is not None:
        # 将传入的文件转为Opencv格式
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        # 展示原图片
        st.image(opencv_image, width=300, channels="BGR")


        res_search = img_search(opencv_image)
        ls_img = []
        ls_dis = []

        for img, dis in res_search:
            img_url = '' + img + '.jpg'
            ls_img.append(img_url)
            dis_s = '此图与原图距离为：' + str(dis)
            ls_dis.append(dis_s)
            st.image(img_url, width=100, caption=dis_s)
        # st.image(ls_img, width=100, caption=ls_dis)












if __name__ == "__main__":
    main()