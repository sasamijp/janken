# coding=utf-8

# get_segment.pyで生成したセグメントを読み込み, 輪郭点の数と結合重みを振って認識率を確かめる実験を行うスクリプト

import numpy as np
import cv2
from matplotlib import pyplot as plt
from collections import Counter


def read_csv(filename):
    return map(lambda line: line.split(','), list(open(filename, 'r')))


def normalize(v):
    m = max(v)
    for i in range(len(v)):
        v[i] = v[i] / m
    return v


def np_hist_to_cv(np_hist):
    counts, bin_edges = np_hist
    return counts.ravel().astype('float32')


# hist_r, hist_tが与えられた時, d_histsのどれに属すかをd_labelsから選んで返す.
def recognize(d_hists_r, d_hists_t, d_labels, hist_r, hist_t, w=0.5, k=5):
    dists = []

    # len(dists) == len(d_hists_r)
    for i in range(len(d_hists_r)):
        dist_r = cv2.compareHist(d_hists_r[i], hist_r, cv2.HISTCMP_BHATTACHARYYA)
        dist_t = cv2.compareHist(d_hists_t[i], hist_t, cv2.HISTCMP_BHATTACHARYYA)
        dists.append(w * dist_r + (1 - w) * dist_t)

    # print dists.index(sorted(dists)[0])

    k_labels = []
    for l in range(k):
        k_labels.append(d_labels[dists.index(sorted(dists)[l])])

        # print d_labels
        # print dists
        # print k_labels

    ans = [ite for ite, it in Counter(k_labels).most_common(1)]
    return ans[0]


def get_hists_from_csv(filepath):
    data = read_csv(filepath)
    hists_r = []
    hists_t = []
    for i, d in enumerate(data):
        ar = map(lambda x: float(x), d)
        vr = []
        vt = []
        for k, x in enumerate(ar):
            if k % 2 == 1:
                vt.append(x)
            else:
                vr.append(x)

        hists_r.append(np_hist_to_cv(np.histogram(normalize(vr), bins=30)))
        hists_t.append(np_hist_to_cv(np.histogram(vt, bins=30, range=(0, 180))))
    return hists_r, hists_t


def recognize_images(filenames, images_label):
    data = read_csv("./train_feature.csv")
    labels = list("cpgcgppccgppcpgcgppccgppcpgcgppccgppcpgcgppccgpp")

    answered = 0

    hists_r = []
    hists_t = []

    for i, d in enumerate(data):
        ar = map(lambda x: float(x), d)
        vr = []
        vt = []
        for k, x in enumerate(ar):
            if k % 2 == 1:
                vt.append(x)
            else:
                vr.append(x)

        hists_r.append(np_hist_to_cv(np.histogram(normalize(vr), bins=30)))
        hists_t.append(np_hist_to_cv(np.histogram(vt, bins=30, range=(0, 180))))

    for i, filename in enumerate(filenames):
        recog_data = get_feature(filename)

        vr = []
        vt = []

        for k, x in enumerate(recog_data):
            vr.append(x[0])
            vt.append(x[1])

        t_hist_r = np_hist_to_cv(np.histogram(normalize(vr), bins=30))
        t_hist_t = np_hist_to_cv(np.histogram(vt, bins=30, range=(0, 180)))

        ans = recognize(hists_r, hists_t, labels, t_hist_r, t_hist_t)
        if ans == images_label[i]:
            answered += 1
    # print answered, len(images_label)
    print answered / float(len(images_label))


def recognize_feature(hists_r, hists_t, test_hists_r, test_hists_t, w=0.35, k=5):
    labels = list("cpgcgppccgppcpgcgppccgppcpgcgppccgppcpgcgppccgpp")
    #labels = list("pgcpppcgcppccpppcgg")
    answered = 0
    anss = []

    for i in range(len(test_hists_r)):
        ans = recognize(hists_r, hists_t, labels, test_hists_r[i], test_hists_t[i], w, k)
        anss.append(ans)
        if ans == images_label[i]:
            answered += 1

    # print answered, len(images_label)
    return answered/ float(len(images_label))
    #print images_label
    #print anss


def get_contour_sets(edge, dd):
    sets = []
    center = get_centroid(edge)

    r_array = []
    r_array_norms = []

    for i in range(len(edge)):
        r_array.append(edge[i] - center)
        r_array_norms.append(np.linalg.norm(r_array[i]))

    r_ave = np.array(r_array_norms).mean()
    d = int(r_ave / dd)

    for i in range(len(edge) - d):
        t = edge[i + d] - edge[i]
        sets.append(np.array([r_array_norms[i], get_degree(r_array[i], t)]))

    return sets


def get_degree(v1, v2):
    return (360 / (2 * np.pi)) * np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


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


def get_hists_from_segment(filepath, d):
    data = read_csv(filepath)

    features = []
    for edge in data:
        e = map(lambda x: int(x), edge)
        ee = []
        for i in range(len(e) - 1):
            if i % 2 == 0:
                ee.append(np.array([e[i], e[i + 1]]))
        features.append(dimension_change_smoothing(get_contour_sets(ee, 5), d))

    hists_r = []
    hists_t = []
    #f, a = plt.subplots(3, 4)
    #a = a.ravel()

    for i, feature in enumerate(features):
        vr = []
        vt = []
        for x in feature:
            vr.append(x[0])
            vt.append(x[1])

       # print normalize(vr)
        #if i < 12:
        #    a[i].hist(normalize(vr), bins=30)
        #    a[i].set_xlim(0, 1)
        #    a[i].set_ylim(0, 25)
        #    a[i].set_title(str(i))

        hists_r.append(np_hist_to_cv(np.histogram(normalize(vr), bins=30)))
        hists_t.append(np_hist_to_cv(np.histogram(vt, bins=30, range=(0, 180))))
    #plt.tight_layout()
    #plt.show()
    return hists_r, hists_t


if __name__ == '__main__':
    # print ','.join(map(lambda v: str(v[0]) + ',' + str(v[1]), feature))

    images_label = list("pgcpppcgcppccpppcgg")
    #images_label = list("cpgcgppccgppcpgcgppccgppcpgcgppccgppcpgcgppccgpp")

    for d in range(1, 16):
        #print "d=" + str(d)

        test_hists_r, test_hists_t = get_hists_from_segment("./test_segments.csv", d*20)
    #exit()

        hists_r, hists_t = get_hists_from_segment("./train_segments.csv", d*20)
    #exit()
        re = []
        for i in range(21):
            #print str(i / 20.0) + ",",
            re.append((recognize_feature(hists_r, hists_t, test_hists_r, test_hists_t, w=i/20.0))),#
            # print ",",
        print np.average(re), np.var(re)
        print re

        # print images_label
        # recognize_feature(w=0.5)

        # recognize_images(file_names, images_label)
