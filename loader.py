from triangle import Triangle, Vertex
from texture import Texture
import jittor as jt


class Loader:
    def __init__(self):
        self.normalList = []
        self.coordList = []
        self.textureCoordList = []
        self.triangleList = []
        self.texture = None

    def load_obj(self, _file_path):
        with open(_file_path, "r") as _f:
            for line in _f.readlines():
                if line.startswith("#"):
                    continue
                elif line.startswith("vn"):
                    _parts = line.split(" ")
                    self.normalList.append(jt.array([float(_parts[1]), float(_parts[2]), float(_parts[3])],
                                                    dtype=jt.float32))
                elif line.startswith("vt"):
                    _parts = line.split(" ")
                    self.textureCoordList.append(jt.array([float(_parts[1]), float(_parts[2])],
                                                          dtype=jt.float32))
                elif line.startswith("v"):
                    _parts = line.split(" ")
                    self.coordList.append(jt.array([float(_parts[1]), float(_parts[2]), float(_parts[3]), 1.0],
                                                   dtype=jt.float32))
                elif line.startswith("f"):
                    _parts = line.split(" ")
                    a = _parts[1].split("/")
                    b = _parts[2].split("/")
                    c = _parts[3].split("/")

                    triangle = Triangle(
                        _a=Vertex(coord=self.coordList[int(a[0]) - 1],
                                  texture_coord=self.textureCoordList[int(a[1]) - 1],
                                  normal=self.normalList[int(a[2]) - 1]),
                        _b=Vertex(coord=self.coordList[int(b[0]) - 1],
                                  texture_coord=self.textureCoordList[int(b[1]) - 1],
                                  normal=self.normalList[int(b[2]) - 1]),
                        _c=Vertex(coord=self.coordList[int(c[0]) - 1],
                                  texture_coord=self.textureCoordList[int(c[1]) - 1],
                                  normal=self.normalList[int(c[2]) - 1])
                    )
                    self.triangleList.append(triangle)

    def load_texture(self, file_path):
        self.texture = Texture(file_path)
        for index in range(len(self.triangleList)):
            self.triangleList[index].texture = self.texture
