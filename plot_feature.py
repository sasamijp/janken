# coding=utf-8

#ヒストグラム出力用のスクリプト

import sys, os
import numpy as np

sys.path.append(os.pardir)

import cv2
from matplotlib import pyplot as plt
import hutil
fig = plt.figure()

data = hutil.read_csv('./f10.csv')
#data.extend(hutil.read_csv('./f2_d90.csv'))
#data.extend(hutil.read_csv('./f2_d180.csv'))
#data.extend(hutil.read_csv('./f2_d270.csv'))

t_data = hutil.read_csv('./f11_t.csv')


def np_hist_to_cv(np_histogram_output):
    counts, bin_edges = np_histogram_output
    return counts.ravel().astype('float32')


hists_r = []
hists_t = []
t_hists_t = t_hists_r = []
f, a = plt.subplots(3, 4)
a = a.ravel()

labels = list("cpgcgppccgpp")
t_labels = list("pgc")

# for i, d in enumerate(data):
#     ar = np.array(map(lambda x: float(x), d))
#     vr = []
#     vt = []
#     for k, x in enumerate(ar):
#         if k % 2 == 1:
#             vt.append(x)
#         else:
#             vr.append(x)
#     #exit()
#
#
#     a[i].hist(vt, bins=30)
#     a[i].set_xlim(0, 180)
#     a[i].set_ylim(0, 25)
#     a[i].set_title(labels[i] + str(i))
#
#     #plt.savefig("../evi/turai1/" + str(i) + ".png")
#     #plt.clf()
#     #hists_r.append(np.histogram()
#     hists_t.append(np.histogram(vt, bins=30, range=(0, 180), normed=True))
# plt.tight_layout()
# plt.show()
# exit()

for i, d in enumerate(t_data):
    ar = np.array(map(lambda x: float(x), d))
    vr = []
    vt = []
    for k, x in enumerate(ar):
        if k % 2 == 1:
            vt.append(x)
        else:
            vr.append(x)

    a[i].hist(hutil.normalize(vr), bins=30)
    a[i].set_xlim(0, 1)
    a[i].set_ylim(0, 25)
    a[i].set_title(t_labels[i] + str(i))

    t_hists_r.append(np.histogram(vr, bins=30, range=(0, 200), normed=True))
    t_hists_t.append(np.histogram(vt, bins=30, range=(0, 180), normed=True))
plt.tight_layout()
plt.show()
exit()
#labels = list("cpgcgppccgppcpgcgppccgppcpgcgppccgppcpgcgppccgpp")


for hist_r in hists_r:
    plt.bar(hists_r)
    plt.show()
    exit()
