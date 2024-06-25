import jittor as jt
import cv2
import numpy as np

# 确保Jittor已正确初始化
jt.flags.use_cuda = 1  # 如果你有GPU并且希望使用GPU加速


class Texture:
    def __init__(self, name):
        # 读取图像数据并转换颜色通道顺序
        self.image_data = cv2.imread(name, cv2.IMREAD_COLOR)
        self.height, self.width, _ = np.shape(self.image_data)

    def get_color(self, _u, _v):
        # 将(u, v)归一化坐标转换为图像像素坐标
        _u = min(_u, 1.0)
        _v = min(_v, 1.0)
        _u = max(_u, 0.0)
        _v = max(_v, 0.0)
        u_img = int(_u * (self.width - 1))
        v_img = int((1 - _v) * (self.height - 1))
        # 获取对应像素的颜色值
        _color = self.image_data[v_img, u_img]
        return jt.array([_color[0], _color[1], _color[2]], dtype=jt.float32)

    def get_color_bilinear(self, _u, _v):
        _u = min(_u, 1.0)
        _v = min(_v, 1.0)
        _u = max(_u, 0.0)
        _v = max(_v, 0.0)

        x1 = int(_u * (self.width - 1))
        x2 = min(x1 + 1, self.width - 1)
        y1 = int((1 - _v) * (self.height - 1))
        y2 = min(y1 + 1, self.height - 1)

        _color = []
        if x1 == x2:
            if y1 == y2:
                _color = self.image_data[x1, y1]
            else:
                _color = self.image_data[x1, y1] * (y2 - _v) + self.image_data[x1, y2] * (_v-y1)
        else:
            if y1 == y2:
                _color = self.image_data[x1, y1] * (x2 - _u) + self.image_data[x2, y1] * (_u-x1)
            else:
                w11 = (x2 - _u) * (y2 - _v)
                w12 = (x2 - _u) * (_v - y1)
                w21 = (_u - x1) * (y2 - _v)
                w22 = (_u - x1) * (_v - y1)
                _color = self.image_data[x1, y1] * w11 + self.image_data[x1, y2] * w12 + \
                         self.image_data[x2, y1] * w21 + self.image_data[x2, y2] * w22
        return jt.array([_color[0], _color[1], _color[2]], dtype=jt.float32)

    def save_image(self, output_name):
        # 将处理后的图像数据保存为文件
        cv2.imwrite(output_name, self.image_data)
