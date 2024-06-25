import os
from pathlib import Path

import jittor as jt
import cv2
import numpy as np


class Render:
    def __init__(self, loader, rst, save_path=None):
        self.rst = rst
        self.save_path = save_path
        if self.save_path is None:
            self.save_path = Path(__file__).parent.__str__() + "/result"
        os.makedirs(self.save_path, exist_ok=True)
        self.loader = loader

    def save_image(self, imageMat, name="output.png"):
        mat = imageMat.numpy()
        cv2.imwrite(self.save_path + "/" + name, mat)

    def projection(self):
        # 透视投影 以及正交变换
        final_matrix = jt.matmul(self.rst.viewport_matrix,self.rst.projection_matrix)
        ll = len(self.loader.triangleList)
        cnt = 0
        for i in range(ll):
            tri = self.loader.triangleList[i]
            if not tri.need_render:
                continue
            cnt += 1
            tri.a.coord = jt.matmul(final_matrix, tri.a.coord)
            tri.b.coord = jt.matmul(final_matrix, tri.b.coord)
            tri.c.coord = jt.matmul(final_matrix, tri.c.coord)
            # 透视除法
            tri.a.div_w()
            tri.b.div_w()
            tri.c.div_w()

    def pre_projection(self):
        # 模型变换，视图变换
        final_matrix = jt.matmul(self.rst.view_matrix, self.rst.model_matrix)
        ll = len(self.loader.triangleList)
        for i in range(ll):
            tri = self.loader.triangleList[i]
            tri.a.coord = jt.matmul(final_matrix, tri.a.coord)
            tri.b.coord = jt.matmul(final_matrix, tri.b.coord)
            tri.c.coord = jt.matmul(final_matrix, tri.c.coord)

    def back_filter(self):
        # 剔出背对的三角形
        return None

    def clipping(self):
        # 裁剪
        ll = len(self.loader.triangleList)
        for i in range(ll):
            tri = self.loader.triangleList[i]
            if not tri.need_render:
                continue
            min_x = min(tri.a.x_(), tri.b.x_(), tri.c.x_())
            max_x = max(tri.a.x_(), tri.b.x_(), tri.c.x_())
            min_y = min(tri.a.y_(), tri.b.y_(), tri.c.y_())
            max_y = max(tri.a.y_(), tri.b.y_(), tri.c.y_())
            if min_x < 0 or min_y < 0 or max_x > self.rst.width or max_y > self.rst.height:
                tri.need_render = False
        return None

    def shader(self):
        self.rst.rasterize_triangle(self.loader.triangleList)
        self.save_image(self.rst.imageMat)

    def render(self):
        self.pre_projection()  # 模型变换，视图变换
        self.back_filter()  # 背向过滤
        self.projection()  # 透视投影
        self.clipping()  # 裁剪
        self.shader()  # 着色
