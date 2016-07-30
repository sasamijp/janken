# coding=utf-8

#画像からの特徴量抽出過程デモ画像生成用のスクリプト

import numpy as np
import cv2

def get_hand_image_bg(imgfile):
    img = cv2.imread(imgfile)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height = img.shape[0]
    width = img.shape[1]
    img = cv2.resize(img, (300, 300 * height / width))
    cv2.imshow("a", img)
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


def near_eight_has_point(img, x, y):
    for i in range(-1, 2):
        for j in range(-1, 2):
            if (i != 0 or j != 0) and img[x + i, y + j] != 0:
                return True
    return False


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
            if img[x, y] == 1 and near_eight_has_point(img, x, y):
                edge.append(np.array([x, y]))
    edge = segmentation(edge)
    return edge


def dimension_change_smoothing(edge, d):
    l = len(edge)
    a = [l / d] * d
    m = l % d
    if m != 0:
        c = 0
        while m > 0:
            a[c] += 1
            m -= 1
            c += 1
            if c > d:
                c = 0
    c = 0
    ret = []
    for i in range(l):
        if i == sum(a[0:c]):
            if 0 < i < l - 1:
                ret.append((edge[i - 1] + edge[i] + edge[i + 1]) / 3.0)
            else:
                ret.append(edge[i])
            c += 1
    return ret


def get_centroid(edge):
    s = np.array([0, 0])
    for v in edge:
        s += v.astype(int)
    return s / len(edge)


def get_degree(v1, v2):
    return (360 / (2 * np.pi)) * np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


def get_contour_sets(edge, img):
    sets = []
    center = get_centroid(edge)

    img_c = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    r_array = []
    r_array_norms = []

    for i in range(len(edge)):
        r_array.append(edge[i] - center)
        r_array_norms.append(np.linalg.norm(r_array[i]))
    for k in range(len(edge)):
        if (k % 10 == 0):
            cv2.line(img_c, (center[1], center[0]), (int(edge[k][1]), int(edge[k][0])), (255, 0, 0), 1)
            #cv2.putText(img_c, str(k), (int(edge[k][1]), int(edge[k][0])), cv2.FONT_HERSHEY_PLAIN, 0.6, (255, 255, 0))

    cv2.imshow("e", img_c)
    r_ave = np.array(r_array_norms).mean()
    d = int(r_ave / 5.0)

    for i in range(len(edge) - d):
        t = edge[i + d] - edge[i]
        sets.append(np.array([r_array_norms[i], get_degree(r_array[i], t)]))

    return sets


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


def get_feature(img_filename):
    img = get_hand_image_bg(img_filename)
    height = img.shape[0]
    width = img.shape[1]
    img = cv2.resize(img, (300, 300 * height / width))

    img = remove_isolated_point(img)

    # cv2.imshow('python' + str(i), img)
    # return

    img = cv2.Laplacian(img, cv2.CV_32F)
    img = all_plus(img)
    edge = get_edge(img)

    feature = dimension_change_smoothing(get_contour_sets(edge), 100)
    return feature


if __name__ == '__main__':
    img = get_hand_image_bg("./test_data/5.JPG")
    cv2.imshow("i", img)

    img = remove_isolated_point(img)
    cv2.imshow("i", img)

    img = cv2.Laplacian(img, cv2.CV_32F)

    cv2.imshow("u", img)

    img = all_plus(img)
    edge = get_edge(img)


    get_contour_sets(edge, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
