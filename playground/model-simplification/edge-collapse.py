import sys
import numpy as np
import heapq
from objloader import OBJ

class Mesh:
    def __init__(self):
        self.V = []
        self.N = []
        self.F = []
        self.FN = []
        
class EdgeCollapse:
    def __init__(self, mesh, reduction_ratio):
        self.reduction_ratio = reduction_ratio
        self.new_vertices = list(mesh.V)
        self.new_faces = list(mesh.F)
        self.new_normals = list(mesh.N) if mesh.N else []
        self.new_face_normals = list(mesh.FN)
        self.edge_queue = []
        self.edge_map = {}

    def compute_edge_costs(self):
        for face in self.new_faces:
            edges = [(face[0], face[1]), (face[1], face[2]), (face[2], face[0])]
            for edge in edges:
                sorted_edge = tuple(sorted(edge))
                if sorted_edge not in self.edge_map:
                    cost = self.calculate_cost(sorted_edge)
                    heapq.heappush(self.edge_queue, (cost, sorted_edge))
                    self.edge_map[sorted_edge] = cost

    def calculate_cost(self, edge):
        v1, v2 = self.new_vertices[edge[0]], self.new_vertices[edge[1]]
        cost = np.linalg.norm(v1 - v2)
        return cost

    def collapse_edge(self, edge):
        v1, v2 = edge
        self.new_vertices[v1] = (self.new_vertices[v1] + self.new_vertices[v2]) / 2
        self.new_vertices[v2] = None

        new_faces = []
        for face in self.new_faces:
            if v2 in face:
                new_face = [v1 if v == v2 else v for v in face]
                if len(set(new_face)) == 3:
                    new_faces.append(new_face)
            else:
                new_faces.append(face)
        self.new_faces = new_faces

    def process(self):
        self.compute_edge_costs()
        target_face_count = int(len(self.new_faces) * self.reduction_ratio)

        while len(self.new_faces) > target_face_count and self.edge_queue:
            cost, edge = heapq.heappop(self.edge_queue)
            if self.new_vertices[edge[0]] is None or self.new_vertices[edge[1]] is None:
                continue

            self.collapse_edge(edge)

        index_mapping = {}
        new_vertices = []
        for i, v in enumerate(self.new_vertices):
            if v is not None:
                index_mapping[i] = len(new_vertices)
                new_vertices.append(v)
        self.new_vertices = new_vertices

        updated_faces = []
        for face in self.new_faces:
            if all(v in index_mapping for v in face):
                updated_face = [index_mapping[v] for v in face]
                if len(set(updated_face)) == 3:
                    updated_faces.append(updated_face)
        self.new_faces = updated_faces

        if not self.new_normals:
            self.new_normals = []
            self.new_face_normals = []
            for face in self.new_faces:
                v0, v1, v2 = [self.new_vertices[v] for v in face]
                normal = np.cross(v1 - v0, v2 - v0)
                if np.any(normal):
                    normal = normal / np.linalg.norm(normal)
                self.new_normals.append(normal)
                self.new_face_normals.append([len(self.new_normals)] * 3)
                
def read_obj(filename):
    mesh = Mesh()
    obj = OBJ(filename, enable_opengl=False)
    
    mesh.V = [np.array(v) for v in obj.vertices]
    mesh.N = [np.array(n) for n in obj.normals] if obj.normals else []
    
    for face in obj.faces:
        vertices, normals, _, _ = face
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
    out_filename = f"{base_name}_ec.obj"
    with open(out_filename, 'w') as f:
        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        
        for n in normals:
            f.write(f"vn {n[0]} {n[1]} {n[2]}\n")
        
        for face, fn in zip(faces, face_normals):
            f.write(f"f {face[0]+1}//{fn[0]} {face[1]+1}//{fn[1]} {face[2]+1}//{fn[2]}\n")
            
def main():
    if len(sys.argv) != 2:
        print("Usage: python script.py <mesh_file>")
        sys.exit(1)

    filename = "Image Synthesis (OpenGL)/DVI2/input-models/" + sys.argv[1]
    outfile = "Image Synthesis (OpenGL)/DVI2/output-models/" + sys.argv[1]
    d = float(input("Enter the reduction ratio: "))

    mesh = read_obj(filename)

    ec = EdgeCollapse(mesh, d)
    ec.process()

    export_obj(outfile, ec.new_vertices, ec.new_faces, ec.new_normals, ec.new_face_normals)

    print(f"Original mesh: {len(mesh.V)} vertices, {len(mesh.F)} faces")
    print(f"Simplified mesh: {len(ec.new_vertices)} vertices, {len(ec.new_faces)} faces")

if __name__ == "__main__":
    main()