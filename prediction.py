# -*- coding:utf-8 -*-
import numpy
import cv2
import sys
import os
import os.path
import re
from progressbar import ProgressBar

from keras.models import Sequential, load_model
from keras.preprocessing.image import load_img, img_to_array
import numpy as np

import argparse
from PIL import Image
from test import *

from print_time import get_time, print_msg


def camera():
    cascade_path = './cascade_file.xml'
    cascade = cv2.CascadeClassifier(cascade_path)
    cap = cv2.VideoCapture(0)
    while True:
        ret, img = cap.read()     
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=2,)
        # for x, y, w, h in faces:
        #     cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        #     face = img[y: y + h, x: x + w]
        #     face_gray = gray[y: y + h, x: x + w]
            
        cv2.imshow('video image', img)
        key = cv2.waitKey(10)
        if len(faces):  # ESCキーで終了
            path = "./src/photo.png"
            cv2.imwrite(path,img)
            break

    cap.release()
    cv2.destroyAllWindows()


def kiritori():

    cascade_path = './cascade_file.xml'
    cascade = cv2.CascadeClassifier(cascade_path)
    # cascade_path = '/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml'
    # cascade_path = '/usr/local/share/OpenCV/haarcascades/lbpcascade_animeface.xml'

    # argvs = sys.argv   # コマンドライン引数を格納したリストの取得
    # argc = len(argvs)  # 引数の個数

    # directory = argv
    # まず元画像があるディレクトリを持ってくる
    image_path = "./src"
    # file_namesにディレクトリ込みのファイルネームを格納
    file_names = filenamelist(image_path)
    # 引数にとった出力先ディレクトリの存在確認 & 作成

    output_path = "./output"
    # create_directory(output_path)

    i = 0

    if len(file_names) == 0:
        print("Not exist jpg or png file !")
    else :
        p = ProgressBar(len(file_names))
        print("File num = ",len(file_names))

    # 元画像のファイルの数だけ顔認識を続ける
    for file_name in file_names:
        # print file_name
        facedetect(file_name, output_path, 1,cascade_path)
        p.update(i+1)
        i += 1

    img = Image.open('./output/photo.png')
    img_resize = img.resize((128, 128))
    img_resize.save('./output/photo.png')

    print("Complete !")


def face_detection_from_path(path, size):

    cv_img = cv2.imread(path)
    cascade_path = "./lib/haarcascade_frontalface_alt.xml"
    cascade = cv2.CascadeClassifier(cascade_path)
    face_rect = cascade.detectMultiScale(cv_img, scaleFactor=1.1, minNeighbors=1, minSize=(1, 1))
    face_data = []
    for rect in face_rect:
        face_img = cv_img[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]
        resized = cv2.resize(face_img, None, fx=float(size/face_img.shape[0]), fy=float(size/face_img.shape[1]))
        cv_im_rgb = resized[:, :, ::-1].copy()
        pil_img = Image.fromarray(cv_im_rgb)
        face_data.append(pil_img)

    return face_data


def identify():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', '-m', default='mymodel.h5')

    # 判定したいファイルの指定

    kiritori()

    # todo; ここのパス指定を，中倉くんが作成してくれたプログラムが保存した画像に変更する
    parser.add_argument('--testpath', '-t', default='./output/photo.png')

    args = parser.parse_args()

    num_classes = 7
    img_rows, img_cols = 128, 128

    ident = [""] * num_classes
    for line in open("who_is_who.txt", "r"):
        dir_name = line.split(",")[0]
        label = line.split(",")[1]
        ident[int(label)] = dir_name

    model = load_model(args.model)

    # 引数 testpath を実際に使用してるのはここ
    face_imgs = face_detection_from_path(args.testpath, img_rows)

    # print(face_imgs)                                            # for debug

    img_array = []
    for face_img in face_imgs:
        # face_img.show()
        img_array.append(img_to_array(face_img))
    img_array = np.array(img_array) / 255.0
    img_array.astype('float32')

    preds = model.predict(img_array, batch_size=img_array.shape[0])
    for pred in preds:
        predR = np.round(pred)
        for pre_i in np.arange(len(predR)):
            if predR[pre_i] == 1:
                detect_name = ident[pre_i]
                # print("he/she is {}".format(detect_name))      # for debug

    return detect_name


def print_result(person_name):
    
    #print("記録")
    print(person_name)
    time_str, now_time = get_time()
    msg = print_msg(now_time)
    print(time_str, msg)


def main():
    camera()
    person_name = identify()
    print_result(person_name)


if __name__ == '__main__':
    main()
