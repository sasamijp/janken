# coding=utf-8

#get_segment.pyで生成したセグメントから特徴量を得るスクリプト

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


def read_csv(filename):
    return map(lambda line: line.split(','), list(open(filename, 'r')))


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


def near_eight_search(edge, x, y):
    for i in range(-1, 2):
        for j in range(-1, 2):
            n = np.where(edge == np.array([x + i, y + j]))[0]
            # print n
            if (i != 0 or j != 0) and len(n) != 0:
                print n[0]
                return n[0]
    return None


def near_eight_search_roll(edge, x, y, img):
    P = [[-1, -1], [0, -1], [1, -1], [-1, 0], [1, 0], [-1, 1], [0, 1], [1, 1]]
    for p in P:
        i = p[0]
        j = p[1]
        if img[x + i, y + j] != 0:
            for k, e in enumerate(edge):
                if e[0] == x + i and e[1] == y + j:
                    return k
    return None


def segmentation(edge):
    ret = []
    copy = edge
    for i in range(len(edge)):
        copy = sorted(copy, key=lambda vv: np.linalg.norm(vv - copy[0]))
        ret.append(copy.pop(0))
    return ret


def segmentation2(edge, img):
    ret = []
    copy = edge
    n = copy[0]

    for i in range(len(edge)):
        near_search = near_eight_search_roll(copy, n[0], n[1], img)
        if near_search is None:
            copy = sorted(copy, key=lambda vv: np.linalg.norm(vv - copy[0]))
            n = copy.pop(0)
            ret.append(n)
            print "!!!!!!!!!!!"
        else:
            n = copy.pop(near_search)
            # print n
            ret.append(n)
    return ret


def segmentation3(edge):
    D = []
    for i in range(len(edge)):
        copy = edge[:]
        del copy[i]
        D.append(map(lambda p: np.linalg.norm(p - edge[i]), copy))
    print D


def get_edge(img):
    # print "searchpoint start"
    edge = []
    for x in range(1, len(img) - 1):
        for y in range(1, len(img[0]) - 1):
            if img[x, y] == 1:  # and near_eight_has_point(img, x, y):
                edge.append(np.array([x, y]))
    # print "searchpoint end"

    edge = segmentation(edge)
    return edge


def get_edge2(img):
    P = [[-1, -1], [0, -1], [1, -1], [-1, 0], [1, 0], [-1, 1], [0, 1], [1, 1]]
    edge = []
    for x in range(1, len(img) - 1):
        for y in range(1, len(img[0]) - 1):
            if img[x, y] == 1:  # and near_eight_has_point(img, x, y):
                edge.append(np.array([x, y]))
    # print "searchpoint end"

    for p in P:
        i = p[0]
        j = p[1]
        if img[x + i, y + j] != 0:
            return np.array([x + i, y + j])

    return None


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


def get_contour_sets(edge):
    sets = []
    center = get_centroid(edge)

    r_array = []
    r_array_norms = []

    for i in range(len(edge)):
        r_array.append(edge[i] - center)
        r_array_norms.append(np.linalg.norm(r_array[i]))

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
    # print "getimage start"
    img = get_hand_image_bg(img_filename)
    height = img.shape[0]
    width = img.shape[1]
    img = cv2.resize(img, (300, 300 * height / width))
    # print "getimage end"

    # print "removeisolated start"
    img = remove_isolated_point(img)
    # print "removeisolated end"

    # ret, thresh = cv2.threshold(img,
    #              0,
    #             255,
    #              cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # c = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]

    # print c[0]
    # cv2.imshow('python' + str(i), img)
    # return
    # print "laplacian start"
    img = cv2.Laplacian(img, cv2.CV_32F)
    img = all_plus(img)
    # print "laplacian end"

    edge = get_edge(img)
    # img.fill(255)
    # print "a ====" + str(len(c[0]))
    # for i, cc in enumerate(c[0]):
    #    print cc[0], cc[len(cc)-1], len(cc)
    #    cv2.drawContours(img, cc, -1, (i*50, 255-i*50, i*50), 3)

    # dedge = dimension_change_smoothing(edge, 100)]
    # dedge = edge
    # for i in range(len(edge) - 1):
    #   print(np.linalg.norm(edge[i] - edge[i + 1]))
    # for k in range(len(dedge)-20):
    #    if (k % 20 == 0):
    #        cv2.line(img, (int(dedge[k+20][1]), int(dedge[k+20][0])), (int(dedge[k][1]), int(dedge[k][0])), (255, 0, 0), 1)
    # cv2.putText(img, str(k), (int(dedge[k][1]), int(dedge[k][0])), cv2.FONT_HERSHEY_PLAIN, 0.6, (255, 255, 0))

    # cv2.imshow(img_filename, img)
    # print "coutour start"
    feature = dimension_change_smoothing(get_contour_sets(edge), 100)
    # print "countour end"
    return feature


if __name__ == '__main__':
    data = read_csv("train_segments.csv")

    for edge in data:
        e = map(lambda x: int(x), edge)
        ee = []
        for i in range(len(e) - 1):
            ee.append(np.array([e[i], e[i+1]]))
        #print ee
        #print get_contour_sets(ee)

        feature = dimension_change_smoothing(get_contour_sets(ee), 100)

        print ','.join(map(lambda v: str(v[0]) + ',' + str(v[1]), feature))

        ##for i in range(2):
        #    data = read_csv()
        #    print ','.join(map(lambda v: str(v[0]) + ',' + str(v[1]), data))
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
