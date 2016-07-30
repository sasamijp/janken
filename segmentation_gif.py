# coding=utf-8

#セグメント並び替えデモ用のgif画像の生成スクリプト

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
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
    #cv2.imshow("niti", img)
    img = cv2.Laplacian(img, cv2.CV_32F)
    img = all_plus(img)

    return img, get_edge(img)


def build_gif(imgs, title=''):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_axis_off()

    ims = map(lambda x: (ax.imshow(x), ax.set_title(title)), imgs)

    im_ani = animation.ArtistAnimation(fig, ims, interval=800, repeat_delay=0, blit=False)

    im_ani.save('animation.gif', writer='imagemagick')


def all_255(img):
    for i in range(len(img)):
        for k in range(len(img[0])):
            if img[i][k] <= 0:
                img[i][k] = 0
            else:
                img[i][k] = 255
    return img


if __name__ == '__main__':
    img, edge = get_segment("../eval/train_data/8.png")
    #cv2.imshow("a", img)

    img = all_255(img)
    #img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for i, p in enumerate(edge):
        cimg = img.copy()

        cimg = cv2.cvtColor(cimg, cv2.COLOR_GRAY2RGB)
        cv2.circle(cimg, (p[1], p[0]), 2, (0, 255, 0), -1)
      #  imgs.append(cimg)
        #cv2.imshow("1", cimg)
        #print cimg[0]

        cv2.imwrite("./animation/" + str(i) + ".png", cimg)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        #exit()
    #build_gif(imgs)#

    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
