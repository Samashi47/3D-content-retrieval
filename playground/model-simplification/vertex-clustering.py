import numpy as np
from objloader import OBJ
import os

class Mesh:
    def __init__(self):
        self.V = []
        self.F = []
        self.N = []
        self.FN = []

class QuadricErrorMetric:
    def __init__(self):
        self.A = np.zeros((3, 3))
        self.b = np.zeros(3)

    def compute(self, weight, c):
        A_dash = self.A + weight * np.eye(3)
        b_dash = self.b + weight * c
        return np.linalg.solve(A_dash, b_dash)

class VertexClustering:
    def __init__(self, mesh, ratio):
        self.V = mesh.V
        self.F = mesh.F
        self.N = mesh.N
        self.FN = mesh.FN
        self.newV = []
        self.newF = []
        self.newN = []
        self.newFN = []
        
        vertices = np.array(self.V)
        bbox_min = np.min(vertices, axis=0)
        bbox_max = np.max(vertices, axis=0)
        bbox_size = bbox_max - bbox_min
        max_dim = np.max(bbox_size)
        self.cell_length = ratio * max_dim
        self.min_coords = bbox_min
        
    def compute(self):

        cell_map = {}
        normal_map = {}
        cell_v_num = []
        newV_idx = 0

        for f, fn in zip(self.F, self.FN):
            cell_idx = []
            for i in range(3):
                v = self.V[f[i]]
                cell = tuple(((v - self.min_coords) / self.cell_length).astype(int))
                cell_idx.append(cell)

            if cell_idx[0] != cell_idx[1] and cell_idx[1] != cell_idx[2] and cell_idx[2] != cell_idx[0]:
                face = []
                for i in range(3):
                    if cell_idx[i] not in cell_map:
                        cell_map[cell_idx[i]] = newV_idx
                        normal_map[cell_idx[i]] = np.zeros(3)
                        newV_idx += 1
                    face.append(cell_map[cell_idx[i]])
                    normal_map[cell_idx[i]] += self.N[fn[i]-1]
                self.newF.append(face)
                self.newFN.append([i+1 for i in face])

        self.newN = []
        for cell in cell_map:
            norm = normal_map[cell]
            if np.any(norm):
                norm = norm / np.linalg.norm(norm)
            else:
                norm = np.array([0.0, 1.0, 0.0])
            self.newN.append(norm)

        cell_v_num = [0] * len(cell_map)
        for i, v in enumerate(self.V):
            cell_idx = tuple(((v - self.min_coords) / self.cell_length).astype(int))
            if cell_idx in cell_map:
                cell_v_num[cell_map[cell_idx]] += 1

        QEM = [QuadricErrorMetric() for _ in range(len(cell_map))]
        for f in self.F:
            v0, v1, v2 = np.array(self.V[f[0]]), np.array(self.V[f[1]]), np.array(self.V[f[2]])
            normal = np.cross(v1 - v0, v2 - v0)
            normal = normal / np.linalg.norm(normal)

            for i in range(3):
                v = self.V[f[i]]
                cell_idx = tuple(((v - self.min_coords) / self.cell_length).astype(int))
                if cell_idx in cell_map:
                    idx = cell_map[cell_idx]
                    A = np.outer(normal, normal)
                    QEM[idx].A += A
                    QEM[idx].b += A @ v

        self.newV = [np.zeros(3) for _ in range(len(cell_map))]
        for cell, idx in cell_map.items():
            cell = np.array(cell)
            c = cell * self.cell_length + self.min_coords + np.array([0.5, 0.5, 0.5]) * self.cell_length
            weight = cell_v_num[idx] * 0.001
            self.newV[idx] = QEM[idx].compute(weight, c)


def read_obj(filename):
    mesh = Mesh()
    obj = OBJ(filename, enable_opengl=False)
    
    mesh.V = [np.array(v) for v in obj.vertices]
    mesh.N = [np.array(n) for n in obj.normals] if obj.normals else []
    
    for face in obj.faces:
        vertices, normals = face
        mesh.F.append([v-1 for v in vertices[:3]])
        mesh.FN.append(normals[:3])
        
        if not mesh.N:
            v0, v1, v2 = [mesh.V[v-1] for v in vertices[:3]]
            normal = np.cross(v1 - v0, v2 - v0)
            if np.any(normal):
                normal = normal / np.linalg.norm(normal)
            mesh.N.append(normal)
            mesh.FN[-1] = [len(mesh.N), len(mesh.N), len(mesh.N)]
    
    return mesh

def export_obj(filename, vertices, faces, normals, face_normals):
    base_name = filename[:-4]
    out_filename = f"{base_name}_vc.obj"
    with open(out_filename, 'w') as f:
        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        
        for n in normals:
            f.write(f"vn {n[0]} {n[1]} {n[2]}\n")
        
        for face, fn in zip(faces, face_normals):
            f.write(f"f {face[0]+1}//{fn[0]} {face[1]+1}//{fn[1]} {face[2]+1}//{fn[2]}\n")

def simpify_db():
    root_folder = "3D Models"
    for category in os.listdir(root_folder):
        category_path = os.path.join(root_folder, category)
        if not os.path.isdir(category_path):
            continue
        if not os.path.exists(os.path.join("benchmark/3D Models S06", category)):
            os.makedirs(os.path.join("benchmark/3D Models S06", category))
            
        for model_name in os.listdir(category_path):
            model_path = os.path.join(category_path, model_name)
            model = OBJ(model_path, enable_opengl=False)
            if model is None:
                continue
            
            print(f"Processing {model_name} from {category}...")
            mesh = read_obj(os.path.join("3D Models", category, model_name))
            d = 0.1
            
            vc = VertexClustering(mesh, d)
            vc.compute()
            outfile = os.path.join("benchmark/3D Models S06",category, model_name)
            export_obj(outfile, vc.newV, vc.newF, vc.newN, vc.newFN)
            print(f"Original mesh: {len(mesh.V)} vertices, {len(mesh.F)} faces")
            print(f"Simplified mesh: {len(vc.newV)} vertices, {len(vc.newF)} faces")
            
if __name__ == "__main__":
    simpify_db()