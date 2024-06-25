"""
1.eye_pos,eye_dir,up_dir,aspect_ratio ,near_plane,far_plane
2.model_projection view_projection
3.Rasterizer<光栅化> + Shader<着色<sampling>>
4.write
"""
import numpy as np
import jittor as jt

from loader import Loader
from rasterizer import Rasterizer
from render import Render

jt.flags.use_cuda = 0


def get_view_matrix(_eye_pos, _eye_dir, _up_dir):
    mov_mat = jt.array([[1.0, 0.0, 0.0, -1 * _eye_pos[0]],
                        [0.0, 1.0, 0.0, -1 * _eye_pos[1]],
                        [0.0, 0.0, 1.0, -1 * _eye_pos[2]],
                        [0.0, 0.0, 0.0, 1.0]])

    _z = jt.array([_eye_dir[0], _eye_dir[1], _eye_dir[2]])  # -z
    _z = _z / jt.norm(_z)  # 归一化
    y = jt.array([_up_dir[0], _up_dir[1], _up_dir[2]])  # y
    y = y / jt.norm(y)  # 归一化
    x = jt.cross(_z, y).flatten()  # x = -z cross y
    R_anti = jt.array([[x[0].item(), y[0].item(), -1 * _z[0].item(), 0.0],
                       [x[1].item(), y[1].item(), -1 * _z[1].item(), 0.0],
                       [x[2].item(), y[2].item(), -1 * _z[2].item(), 0.0],
                       [0.0, 0.0, 0.0, 1.0]])
    R = R_anti.transpose()  # 正交矩阵的逆矩阵为其转置

    return jt.matmul(R, mov_mat)


def get_model_matrix(mov=None, rotate_angle_x=0.0, rotate_angle_y=0.0, rotate_angle_z=0.0, scale=1.0):
    # 弧度制
    if mov is None:
        mov = [0, 0, 0]
    translation_matrix = jt.array([
        [1, 0, 0, mov[0]],
        [0, 1, 0, mov[1]],
        [0, 0, 1, mov[2]],
        [0, 0, 0, 1]
    ], dtype=jt.float32)
    c = jt.cos(jt.array(rotate_angle_x)).item()
    s = jt.sin(jt.array(rotate_angle_x)).item()
    rotate_x_matrix = jt.array([
        [1.0, 0, 0, 0],
        [0, c, s * -1, 0],
        [0, s, c, 0],
        [0, 0, 0, 1]
    ], dtype=jt.float32)
    c = jt.cos(jt.array(rotate_angle_y)).item()
    s = jt.sin(jt.array(rotate_angle_y)).item()
    rotate_y_matrix = jt.array([
        [c, 0, s, 0],
        [0, 1, 0, 0],
        [-1 * s, 0, c, 0],
        [0, 0, 0, 1]
    ], dtype=jt.float32)
    c = jt.cos(jt.array(rotate_angle_z)).item()
    s = jt.sin(jt.array(rotate_angle_z)).item()
    rotate_z_matrix = jt.array([
        [c, -1 * s, 0, 0],
        [s, c, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=jt.float32)
    scale_matrix = jt.array([
        [scale, 0, 0, 0],
        [0, scale, 0, 0],
        [0, 0, scale, 0],
        [0, 0, 0, 1]
    ], dtype=jt.float32)
    return jt.matmul(translation_matrix,
                     jt.matmul(rotate_x_matrix, jt.matmul(rotate_y_matrix, jt.matmul(rotate_z_matrix,
                                                                                     scale_matrix))))


def main():
    obj_path = "obj/cow/cow.obj"
    texture_path = "obj/cow/cow_texture.png"
    my_loader = Loader()
    my_loader.load_obj(obj_path)  # 先加载模型
    my_loader.load_texture(texture_path)  # 再加载纹理
    eye_pos = [0, 0, 0]  # 相机位置
    eye_dir = [0, 0, 1]  # 视线
    up_dir = [0, 1, 0]  # 上方向
    horizontal_angle = (130.0 / 360) * 2 * np.pi  # 水平视场角
    aspect_ratio = 1.0  # 宽高比
    scale_ratio = 2.6  # 缩放
    obj_pose = [0, 0, 3]  # 放置位置

    # 视图变换
    view_matrix = get_view_matrix(_eye_pos=eye_pos, _eye_dir=eye_dir, _up_dir=up_dir)
    # 模型变换
    model_matrix = get_model_matrix(mov=obj_pose, scale=scale_ratio, rotate_angle_z=np.pi / 2)

    # 设置投影矩阵
    rst = Rasterizer(700, 700)
    rst.set_view_matrix(view_matrix)
    rst.set_model_matrix(model_matrix)
    rst.set_projection_matrix(horizontal_angle=horizontal_angle, aspect_ratio=aspect_ratio, zNear=2, zFar=50)
    my_render = Render(rst=rst, save_path="result", loader=my_loader)
    my_render.render()
    return None


if __name__ == "__main__":
    main()
