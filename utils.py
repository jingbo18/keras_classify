#!/usr/bin/env python
# coding: utf-8

import os
import cv2

__all__ = ["mkdir", "save_data_to_png"]


def mkdir(dir):
    if not os.path.exists(dir):
        print("mkdir: ", dir)
        return os.makedirs(dir)


def save_data_to_png(data, dir):
    (d, h, w) = data.shape
    mkdir(dir)
    for i in list(range(d)):
        cv2.imwrite(os.path.join(dir, str(i) + ".png"), data[i])
