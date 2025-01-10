import numpy as np


def voxelize_model(vertices: np.ndarray, faces: np.ndarray, N: int = 128) -> np.ndarray:
    voxels = np.zeros((N, N, N))

    min_coords = np.min(vertices, axis=0)
    max_coords = np.max(vertices, axis=0)
    cube_size = max(max_coords - min_coords)

    vertices = (vertices - min_coords) / cube_size - 0.5

    voxel_coords = ((vertices + 0.5) * N).astype(int)
    voxel_coords = np.clip(voxel_coords, 0, N - 1)

    for face in faces:
        v1, v2, v3 = voxel_coords[[i - 1 for i in face]]
        min_x = max(0, min(v1[0], v2[0], v3[0]))
        max_x = min(N - 1, max(v1[0], v2[0], v3[0]))
        min_y = max(0, min(v1[1], v2[1], v3[1]))
        max_y = min(N - 1, max(v1[1], v2[1], v3[1]))
        min_z = max(0, min(v1[2], v2[2], v3[2]))
        max_z = min(N - 1, max(v1[2], v2[2], v3[2]))

        for x in range(min_x, max_x + 1):
            for y in range(min_y, max_y + 1):
                for z in range(min_z, max_z + 1):
                    voxels[x, y, z] = 1

    return voxels


def compute_3d_dft(voxels: np.ndarray, N: int = 128, K: int = 2) -> np.ndarray:
    fourier = np.fft.fftn(voxels)
    fourier = np.fft.fftshift(fourier)

    center = N // 2
    K_range = range(-K, K + 1)
    coeffs = []

    for u in K_range:
        for v in K_range:
            for w in K_range:
                coeff = fourier[center + u, center + v, center + w]
                if u > 0 or (u == 0 and (v > 0 or (v == 0 and w >= 0))):
                    coeffs.append(np.abs(coeff))

    return np.array(coeffs)


def compute_descriptor(model: object, N: int = 128, K: int = 2) -> np.ndarray:
    vertices = np.array(model.vertices)
    faces = np.array(model.faces)
    voxels = voxelize_model(vertices, faces, N)

    feature_vector = compute_3d_dft(voxels, N, K)

    return feature_vector


def compute_distance_fourier(
    desc1: np.ndarray, desc2: np.ndarray, metric: str = "l1"
) -> float:
    if metric == "l1":
        return np.sum(np.abs(desc1 - desc2))
    elif metric == "l2":
        return np.sqrt(np.sum((desc1 - desc2) ** 2))
    else:
        raise ValueError("Unsupported metric. Use 'l1' or 'l2'.")
