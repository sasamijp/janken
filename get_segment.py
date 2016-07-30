# coding=utf-8

# 画像を読み込んでセグメント(接続順に並んだ輪郭点)を出力するスクリプト

import numpy as np
import cv2


def get_hand_image_bg(imgfile):
    img = cv2.imread(imgfile)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, img = cv2.threshold(img,
                             200,
                             255,
                             cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return img


def all_plus(img):
    for i in range(len(img)):
        for k in range(len(img[0])):
            if img[i][k] <= 0:
                img[i][k] = 0
            else:
                img[i][k] = 1
    return img


def segmentation(edge):
    ret = []
    copy = edge
    for i in range(len(edge)):
        copy = sorted(copy, key=lambda vv: np.linalg.norm(vv - copy[0]))
        ret.append(copy.pop(0))
    return ret


def get_edge(img):
    edge = []
    for x in range(1, len(img) - 1):
        for y in range(1, len(img[0]) - 1):
            if img[x, y] == 1:
                edge.append(np.array([x, y]))
    return segmentation(edge)


def get_degree(v1, v2):
    return (360 / (2 * np.pi)) * np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


def remove_isolated_point(img):
    n8 = np.array([[1, 1, 1],
                   [1, 1, 1],
                   [1, 1, 1]],
                  np.uint8)
    img = cv2.erode(img,
                    n8,
                    iterations=1)
    return cv2.dilate(img,
                      n8,
                      iterations=1)


def get_segment(img_filename):
    img = get_hand_image_bg(img_filename)
    height = img.shape[0]
    width = img.shape[1]
    img = cv2.resize(img, (300, 300 * height / width))

    img = remove_isolated_point(img)
    img = cv2.Laplacian(img, cv2.CV_32F)
    img = all_plus(img)

    return get_edge(img)


if __name__ == '__main__':
    for i in range(20):
        #data = get_feature("./train_data/" + str(i) + ".png")
        edge = get_segment("./test_data/" + str(i) + ".JPG")
        #print ','.join(map(lambda v: str(v[0]) + ',' + str(v[1]), data))
        print ','.join(map(lambda x: str(x[0]) +","+str(x[1]), edge))
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
