import scipy.misc
import numpy as np
import os

import sys


def save_img(out_path, img):
    img = np.clip(img, 0, 255).astype(np.uint8)
    scipy.misc.imsave(out_path, img)


def scale_img(style_path, style_scale):
    scale = float(style_scale)
    o0, o1, o2 = scipy.misc.imread(style_path, mode='RGB').shape
    # scale = float(style_scale),重复注释命令行.
    new_shape = (int(o0 * scale), int(o1 * scale), o2)
    style_target = get_img(style_path, img_size=new_shape)
    return style_target


def get_img(src, img_size=False):
    img = scipy.misc.imread(src, mode='RGB')  # misc.imresize(, (256, 256, 3))
    if not (len(img.shape) == 3 and img.shape[2] == 3):
        img = np.dstack((img, img, img))
    if img_size != False:
        img = scipy.misc.imresize(img, img_size)
    return img


def exists(p, msg):
    assert os.path.exists(p), msg


def list_files(in_path):
    files = []
    for (dirpath, dirnames, filenames) in os.walk(in_path):

        """
        理解函数.
        dirpath = "F:\data\celebA"
        dirnames = []
        filenames = ['188500.jpg', '188516.jpg', '188517.jpg', '188518.jpg']
        貌似列表已经创建，为什么还需要利用extend创建新的列表.
        """

        files.extend(filenames)
        # break

    return files


if __name__ == '__main__':
    files = list_files("F:\data\celebA")
    print(str(type(files)) + "\n" + str(files))
