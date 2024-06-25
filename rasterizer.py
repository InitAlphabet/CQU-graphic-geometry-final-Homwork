import numpy as np
import jittor as jt


class Rasterizer:
    def __init__(self, width=700, height=700):
        self.zBuffer = -1 * np.ones((width, height)) * 7777
        self.zNear = 1
        self.zFar = 50
        self.aspect_ratio = 1
        self.horizontal_angle = np.pi / 2
        self.view_matrix = None
        self.model_matrix = None
        self.projection_matrix = None
        self.width = width  # 像素
        self.height = height  # 像素
        self.imageMat = jt.zeros((self.width, self.height, 3))
        # 视口变换
        self.viewport_matrix = jt.array([[self.width / 2, 0.0, 0., self.width / 2],
                                         [0., self.height / 2, 0., self.height / 2],
                                         [0., 0., 1., 0.],
                                         [0., 0., 0., 1.]])

    def set_color(self, x, y, color):
        x = self.height - int(x)
        y = int(y)
        self.imageMat[x][y] = color

    def clear(self):
        self.zBuffer = -1 * np.ones((self.width, self.height)) * 7777

    def set_zBuffer(self, x, y, z):
        self.zBuffer[int(x)][int(y)] = z

    def get_zBuffer(self, x, y):
        return self.zBuffer[int(x)][int(y)]

    def set_view_matrix(self, view_matrix):
        self.view_matrix = view_matrix

    def set_model_matrix(self, model_matrix):
        self.model_matrix = model_matrix

    def set_projection_matrix(self, horizontal_angle=np.pi / 2, aspect_ratio=1, zNear=1, zFar=50):
        self.horizontal_angle = horizontal_angle
        self.aspect_ratio = aspect_ratio
        self.zNear = -zNear  # 最后会面向 z负轴
        self.zFar = -zFar  #
        # l = -1 * zNear * (jt.tan(jt.array(horizontal_angle / 2))).item()
        r = zNear * (jt.tan(jt.array(horizontal_angle / 2))).item()
        # b = l * aspect_ratio
        t = r * aspect_ratio
        self.projection_matrix = jt.array([[self.zNear / r, 0, 0, 0],
                                           [0, self.zNear / t, 0, 0],
                                           [0, 0, (self.zNear + self.zFar) / (self.zNear - self.zFar),
                                            -2 * self.zFar * self.zNear / (self.zNear - self.zFar)],
                                           [0, 0, 1, 0]])

    def rasterize_triangle(self, triangleList):
        print("start shader!" + "*" * 10)
        ll = len(triangleList)
        cnt = 0
        for tri in triangleList:
            print(cnt, "/", ll)
            cnt += 1
            if not tri.need_render:
                continue

            min_x = min(tri.a.x_(), tri.b.x_(), tri.c.x_())
            min_x = int(min_x)
            max_x = max(tri.a.x_(), tri.b.x_(), tri.c.x_())
            max_x = int(max_x)
            min_y = min(tri.a.y_(), tri.b.y_(), tri.c.y_())
            min_y = int(min_y)
            max_y = max(tri.a.y_(), tri.b.y_(), tri.c.y_())
            max_y = int(max_y)
            for x in range(min_x, max_x):
                for y in range(min_y, max_y):
                    color = [0, 0, 0]
                    a_inter_z = 0
                    tmp_num = 0
                    for t1 in (0.25, 0.75):
                        for t2 in (0.25, 0.75):
                            x1 = x + t1
                            y1 = y + t2

                            if tri.inTriangle(x1, y1):
                                alpha, beta, gamma = tri.z_interpolation(x1, y1)
                                inter_z = tri.a.w_() * alpha + tri.b.w_() * beta + tri.c.w_() * gamma
                                if inter_z < self.get_zBuffer(x, y):
                                    continue
                                tmp_num += 1
                                inter_texture_coord = alpha * tri.a.texture_coord + \
                                                      beta * tri.b.texture_coord + gamma * tri.c.texture_coord
                                color = color + tri.texture.get_color(inter_texture_coord[0].item(),
                                                                      inter_texture_coord[1].item())
                                a_inter_z = a_inter_z + inter_z
                            if tmp_num > 0:
                                self.set_zBuffer(x, y, a_inter_z // tmp_num)
                                self.set_color(x, y, color / tmp_num)
                    # if tri.inTriangle(x, y):
                    #     alpha, beta, gamma = tri.z_interpolation(x, y)
                    #     inter_z = tri.a.w_() * alpha + tri.b.w_() * beta + tri.c.w_() * gamma
                    #     if inter_z < self.get_zBuffer(x, y):
                    #         continue
                    #     inter_texture_coord = alpha * tri.a.texture_coord + \
                    #                           beta * tri.b.texture_coord + gamma * tri.c.texture_coord
                    #     self.set_zBuffer(x, y, inter_z)
                    #     self.set_color(x, y, tri.texture.get_color(inter_texture_coord[0].item(),
                    #                                                inter_texture_coord[1].item()))
