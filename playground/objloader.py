from OpenGL.GL import *
class OBJ:
    def __init__(self, filename, swapyz=False, enable_opengl=True):
        """Loads a Wavefront OBJ file. """
        self.vertices = []
        self.normals = []
        self.texcoords = []
        self.faces = []
        self.enable_opengl = enable_opengl

        material = None
        for line in open(filename, "r"):
            if line.startswith('#'):
                continue
            values = line.split()
            if not values:
                continue
            if values[0] == 'v':
                v = list(map(float, values[1:4]))
                if swapyz:
                    v = v[0], v[2], v[1]
                self.vertices.append(v)
            elif values[0] == 'vn':
                v = list(map(float, values[1:4]))
                if swapyz:
                    v = v[0], v[2], v[1]
                self.normals.append(v)
            elif values[0] == 'vt':
                self.texcoords.append(list(map(float, values[1:3])))
            elif values[0] in ('usemtl', 'usemat'):
                material = values[1]
            elif values[0] == 'f':
                face = []
                texcoords = []
                norms = []
                for v in values[1:]:
                    w = v.split('/')
                    face.append(int(w[0]))
                    if len(w) >= 2 and len(w[1]) > 0:
                        texcoords.append(int(w[1]))
                    else:
                        texcoords.append(0)
                    if len(w) >= 3 and len(w[2]) > 0:
                        norms.append(int(w[2]))
                    else:
                        norms.append(0)
                if enable_opengl == True:
                    self.faces.append((face, norms, texcoords, material))
                else:
                    if len(face) == 3:
                        self.faces.append(face)
                    elif len(face) == 4:
                        self.faces.append([face[0], face[1], face[2]])
                        self.faces.append([face[0], face[2], face[3]])
                    

        if self.enable_opengl:
            self.gl_list = glGenLists(1)
            glNewList(self.gl_list, GL_COMPILE)
            glEnable(GL_TEXTURE_2D)
            glFrontFace(GL_CCW)
            for face in self.faces:
                vertices, normals, texture_coords, material = face

                glColor(0.8, 0.8, 0.8)

                glBegin(GL_POLYGON)
                for i in range(len(vertices)):
                    if normals[i] > 0:
                        glNormal3fv(self.normals[normals[i] - 1])
                    if texture_coords[i] > 0:
                        glTexCoord2fv(self.texcoords[texture_coords[i] - 1])
                    glVertex3fv(self.vertices[vertices[i] - 1])
                glEnd()
            glDisable(GL_TEXTURE_2D)
            glEndList()