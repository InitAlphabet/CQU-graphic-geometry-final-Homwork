import jittor as jt
import numpy as np


class Vertex:
    def __init__(self, coord=jt.ones((1, 4)), normal=jt.ones((1, 3)), texture_coord=jt.ones((1, 2))):
        # 坐标
        self.coord = coord
        # 法向量
        self.normal = normal
        # 纹理坐标
        self.texture_coord = texture_coord

    def x_(self):
        return self.coord[0].item()

    def y_(self):
        return self.coord[1].item()

    def z_(self):
        return self.coord[2].item()

    def w_(self):
        return self.coord[3].item()

    def get_v3coord(self):
        tmp = self.coord.clone()
        return jt.array([tmp[0].item(), tmp[1].item(), tmp[2].item()],
                        dtype=jt.float32)

    def div_w(self):
        _v = self.coord[3].item()
        self.coord[0] = self.coord[0] / _v
        self.coord[1] = self.coord[1] / _v
        self.coord[2] = self.coord[2] / _v

    def mul_w(self):
        _v = self.coord[3].item()
        self.coord[0] = self.coord[0] * _v
        self.coord[1] = self.coord[1] * _v
        self.coord[2] = self.coord[2] * _v


class Triangle:
    def __init__(self, _a, _b, _c, _texture=None):
        # 顶点
        self.a = _a
        self.b = _b
        self.c = _c
        # 纹理
        self.texture = _texture
        self.need_render = True

    def inTriangle(self, x, y):
        p = jt.array([x, y, 1.0])
        a1 = self.a.get_v3coord()
        b1 = self.b.get_v3coord()
        c1 = self.c.get_v3coord()
        cross1 = jt.cross(b1 - a1, p - a1).flatten()
        cross2 = jt.cross(c1 - b1, p - b1).flatten()
        cross3 = jt.cross(a1 - c1, p - c1).flatten()
        return jt.matmul(cross1, cross2).item() > 0 and jt.matmul(cross1, cross3).item() > 0

    def z_interpolation(self, x, y):
        # 重心插值
        beta = ((y - self.a.y_()) * (self.c.x_() - self.a.x_()) - (x - self.a.x_()) * (self.c.y_() - self.a.y_())) / \
               ((self.b.y_() - self.a.y_()) * (self.c.x_() - self.a.x_()) - (self.b.x_() - self.a.x_()) * (
                       self.c.y_() - self.a.y_()))
        gamma = ((y - self.a.y_()) * (self.b.x_() - self.a.x_()) - (x - self.a.x_()) * (self.b.y_() - self.a.y_())) / \
                ((self.c.y_() - self.a.y_()) * (self.b.x_() - self.a.x_()) - (self.c.x_() - self.a.x_()) * (
                        self.b.y_() - self.a.y_()))
        alpha = 1 - beta - gamma
        z = 1 / (alpha / self.a.w_() + beta / self.b.w_() + gamma / self.c.w_())

        return z * alpha / self.a.w_(), z * beta / self.b.w_(), z * gamma / self.c.w_()

    def is_danger(self):
        tmp = (self.b.y_() - self.c.y_()) * (self.a.x_() - self.c.x_()) + (self.c.y_() - self.a.y_()) * (
                self.b.x_() - self.c.x_())
        return np.isclose(tmp, 0)
