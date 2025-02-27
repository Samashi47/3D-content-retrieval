# Description: This file contains the logic to load the obj file.
class OBJ:
    def __init__(self, file, swapyz=False):
        """Loads a Wavefront OBJ file."""
        self.vertices = []
        self.normals = []
        self.texcoords = []
        self.faces = []

        material = None
        for line in file:
            line = str(line, "utf-8")
            if line.startswith("#"):
                continue
            values = line.split()
            if not values:
                continue
            if values[0] == "v":
                v = list(map(float, values[1:4]))
                if swapyz:
                    v = v[0], v[2], v[1]
                self.vertices.append(v)
            elif values[0] == "vn":
                v = list(map(float, values[1:4]))
                if swapyz:
                    v = v[0], v[2], v[1]
                self.normals.append(v)
            elif values[0] == "vt":
                self.texcoords.append(list(map(float, values[1:3])))
            elif values[0] in ("usemtl", "usemat"):
                material = values[1]
            elif values[0] == "f":
                face = []
                texcoords = []
                norms = []
                for v in values[1:]:
                    w = v.split("/")
                    face.append(int(w[0]))
                    if len(w) >= 2 and len(w[1]) > 0:
                        texcoords.append(int(w[1]))
                    else:
                        texcoords.append(0)
                    if len(w) >= 3 and len(w[2]) > 0:
                        norms.append(int(w[2]))
                    else:
                        norms.append(0)
                if len(face) == 3:
                    self.faces.append(face)
                elif len(face) == 4:
                    self.faces.append([face[0], face[1], face[2]])
                    self.faces.append([face[0], face[2], face[3]])
