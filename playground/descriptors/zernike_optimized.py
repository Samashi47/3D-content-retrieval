import numpy as np
import os
import vtk
import multiprocessing as mp
from scipy.special import (factorial, comb as nchoosek)

IMAG_CONST = 1j
PI_CONST = np.pi
NAN_CONST = np.nan
    
class Pipeline(object):

    def geometric_moments_approx(self, points_array, faces_array, N):
        raise NotImplementedError()

    def geometric_moments_exact(self, points_array, faces_array, N):
        raise NotImplementedError()

def nested_loop(stack, args):
    if len(stack) != 0:
        fn = stack.pop()
        for i in fn(*args):
            for j in nested_loop(stack, args+[i]):
                yield (i,)+j
        stack.append(fn)
    else:
        yield tuple()

def nest(*_stack):
    return nested_loop(list(reversed(_stack)), [])

def autocat(arrs, **dargs):
    axis = dargs.pop('axis', None)
    if axis is None:
        return np.concatenate(arrs, **dargs)
    ndim = arrs[0].ndim
    assert all([ a.ndim == ndim for a in arrs])
    if axis >= ndim:
        arrs = tuple([ np.expand_dims(a,axis) for a in arrs ])
    return np.concatenate(arrs, axis=axis)

class KoehlOptimizations(Pipeline):

    def geometric_moments_exact(self, points_array, faces_array, N):
        n_facets, n_vertices = faces_array.shape[:2]
        assert n_vertices == 3
        moments_array = np.zeros([N+1, N+1, N+1])
        for face in faces_array:
            vertex_list = [points_array[_i, ...] for _i in face]
            moments_array += self.facet_contribution(vertex_list, N)
        return self.factorial_scalar(N) * moments_array

    def facet_contribution(self, vertex_list, N):
        Vf = self.facet_volume(vertex_list)
        Cf = self.term_Cijk(vertex_list[2], N)
        Df = self.term_Dijk(vertex_list[1], N, Cf)
        return Vf*self.term_Sijk(vertex_list[0], N, Df)

    def term_Cijk(self, vertex, N):
        return self.work_loop(vertex, N)

    def term_Dijk(self, vertex, N, Cijk):
        return self.work_loop(vertex, N, Cijk)

    def term_Sijk(self, vertex, N, Dijk):
        return self.work_loop(vertex, N, Dijk)

    def work_loop(self, vertex, N, prev=None):
        R = prev
        if R is None:
            R = np.zeros([N+1, N+1, N+1])
        Q = np.zeros([N+1, N+1, N+1])
        Q[0, 0, 0] = 1.0

        recursion_term = lambda _X, x_y_z, mask: \
            np.roll(_X, 1, axis=0)[mask]*x_y_z[0] + \
            np.roll(_X, 1, axis=1)[mask]*x_y_z[1] + \
            np.roll(_X, 1, axis=2)[mask]*x_y_z[2]
        i, j, k = np.mgrid[:N+1, :N+1, :N+1]
        order = (i+j+k)
        for n in range(N):
            mask = (order==n+1)
            _Q = recursion_term(Q, vertex, mask)
            Q[mask] = _Q + R[mask]
        return Q

def _kmp_geometric_moments_exact_worker(self, vertex_list, N):
    return self.facet_contribution(vertex_list, N)

class KoehlMultiproc(KoehlOptimizations):
    def geometric_moments_exact(self, points_array, faces_array, N):
        n_vertices = faces_array.shape[1]
        assert n_vertices == 3
        moments_array = np.zeros([N+1, N+1, N+1])
        process_pool = mp.Pool()
        for face in faces_array:
            if any(_i >= points_array.shape[0] for _i in face):
                raise IndexError("Index out of bounds for points_array")
            vertex_list = [points_array[_i, ...] for _i in face]
            process_pool.apply_async(_kmp_geometric_moments_exact_worker,
                                     args=(self, vertex_list, N),
                                     callback=moments_array.__iadd__,
                                     )
        process_pool.close()
        process_pool.join()
        return self.factorial_scalar(N) * moments_array
    
class Pipeline(object):

    def geometric_moments_approx(self, points_array, faces_array, N):
        raise NotImplementedError()

    def geometric_moments_exact(self, points_array, faces_array, N):
        raise NotImplementedError()

class SerialPipeline(Pipeline):

    def geometric_moments_exact(self, points_array, faces_array, N):
        n_facets, n_vertices = faces_array.shape[:2]
        assert n_vertices == 3
        moments_array = np.zeros([N + 1, N + 1, N + 1])
        monomial_array = self.monomial_precalc(points_array, N)
        for face in faces_array:
            vertex_list = [points_array[_i, ...] for _i in face]
            Cf_list = [monomial_array[_i, ...] for _i in face]
            Vf = self.facet_volume(vertex_list)
            moments_array += Vf * self.term_Sijk(Cf_list, N)
        return self.factorial_scalar(N) * moments_array

    def factorial_scalar(self, N):
        i, j, k = np.mgrid[0:N + 1, 0:N + 1, 0:N + 1]
        return factorial(i) * factorial(j) * factorial(k) / (factorial(i + j + k + 2) * (i + j + k + 3))

    def monomial_precalc(self, points_array, N):
        n_points = points_array.shape[0]
        monomial_array = np.zeros([n_points, N + 1, N + 1, N + 1])
        tri_array = self.trinomial_precalc(N)
        for point_indx, point in enumerate(points_array):
            monomial_array[point_indx, ...] = self.mon_comb(
                point, tri_array, N)
        return monomial_array

    def mon_comb(self, vertex, tri_array, N, out=None):
        x, y, z = vertex
        c = np.zeros([N + 1, N + 1, N + 1])
        for i, j, k in nest(lambda: range(N + 1),
                            lambda _i: range(N - _i + 1),
                            lambda _i, _j: range(N - _i - _j + 1),
                            ):
            c[i, j, k] = tri_array[i, j, k] * \
                np.power(x, i) * np.power(y, j) * np.power(z, k)
        return c

    def term_Sijk(self, Cf_list, N):
        S = np.zeros([N + 1, N + 1, N + 1])
        C0, C1, C2 = Cf_list
        Dabc = self.term_Dabc(C1, C2, N)
        for i, j, k, ii, jj, kk in nest(lambda: range(N + 1),
                                        lambda _i: range(N - _i + 1),
                                        lambda _i, _j: range(N - _i - _j + 1),
                                        lambda _i, _j, _k: range(_i + 1),
                                        lambda _i, _j, _k, _ii: range(_j + 1),
                                        lambda _i, _j, _k, _ii, _jj: range(
                                            _k + 1),
                                        ):
            S[i, j, k] += C0[ii, jj, kk] * Dabc[i - ii, j - jj, k - kk]
        return S

    def trinomial_precalc(self, N):
        tri_array = np.zeros([N + 1, N + 1, N + 1])
        for i, j, k in nest(lambda: range(N + 1),
                            lambda _i: range(N - _i + 1),
                            lambda _i, _j: range(N - _i - _j + 1)
                            ):
            tri_array[i, j, k] = self.trinomial(i, j, k)
        return tri_array

    def trinomial(self, i, j, k):
        return factorial(i + j + k) / (factorial(i) * factorial(j) * factorial(k))

    def facet_volume(self, vertex_list):
        return np.linalg.det(autocat(vertex_list, axis=1))

    def term_Dabc(self, C1, C2, N):
        D = np.zeros([N + 1, N + 1, N + 1])
        for i, j, k, ii, jj, kk in nest(lambda: range(N + 1),
                                        lambda _i: range(N + 1),
                                        lambda _i, _j: range(N + 1),
                                        lambda _i, _j, _k: range(_i + 1),
                                        lambda _i, _j, _k, _ii: range(_j + 1),
                                        lambda _i, _j, _k, _ii, _jj: range(
                                            _k + 1)
                                        ):
            D[i, j, k] += C1[ii, jj, kk] * C2[i - ii, j - jj, k - kk]
        return D

    def zernike(self, G, N):
        V = np.zeros([N + 1, N + 1, N + 1], dtype=complex)
        for a, b, c, alpha in nest(lambda: range(int(N / 2) + 1),
                                   lambda _a: range(N - 2 * _a + 1),
                                   lambda _a, _b: range(N - 2 * _a - _b + 1),
                                   lambda _a, _b, _c: range(_a + _c + 1),
                                   ):
            V[a, b, c] += np.power(IMAG_CONST, alpha) * \
                nchoosek(a + c, alpha) * G[2 * a + c - alpha, alpha, b]

        W = np.zeros([N + 1, N + 1, N + 1], dtype=complex)
        for a, b, c, alpha in nest(lambda: range(int(N / 2) + 1),
                                   lambda _a: range(N - 2 * _a + 1),
                                   lambda _a, _b: range(N - 2 * _a - _b + 1),
                                   lambda _a, _b, _c: range(_a + 1),
                                   ):
            W[a, b, c] += np.power(-1, alpha) * np.power(2, a - alpha) * \
                nchoosek(a, alpha) * V[a - alpha, b, c + 2 * alpha]

        X = np.zeros([N + 1, N + 1, N + 1], dtype=complex)
        for a, b, c, alpha in nest(lambda: range(int(N / 2) + 1),
                                   lambda _a: range(N - 2 * _a + 1),
                                   lambda _a, _b: range(N - 2 * _a - _b + 1),
                                   lambda _a, _b, _c: range(_a + 1),
                                   ):
            X[a, b, c] += nchoosek(a, alpha) * W[a - alpha, b + 2 * alpha, c]

        Y = np.zeros([N + 1, N + 1, N + 1], dtype=complex)
        for l, nu, m, j in nest(lambda: range(N + 1),
                                lambda _l: range(int((N - _l) / 2) + 1),
                                lambda _l, _nu: range(_l + 1),
                                lambda _l, _nu, _m: range(int((_l - _m) / 2) + 1),
                                ):
            Y[l, nu, m] += self.Yljm(l, j, m) * X[nu + j, l - m - 2 * j, m]

        Z = np.zeros([N + 1, N + 1, N + 1], dtype=complex)
        for n, l, m, nu, in nest(lambda: range(N + 1),
                                 lambda _n: range(_n + 1),
                                 # there's an if...mod missing in this but it
                                 # still works?
                                 lambda _n, _l: range(_l + 1),
                                 lambda _n, _l, _m: range(int((_n - _l) / 2) + 1),
                                 ):
            # integer required for k when used as power in Qklnu below:
            k = int((n - l) / 2)
            Z[n, l, m] += (3 / (4 * PI_CONST)) * \
                self.Qklnu(k, l, nu) * np.conj(Y[l, nu, m])

        for n, l, m in nest(lambda: range(N + 1),
                            lambda _n: range(n + 1),
                            lambda _n, _l: range(l + 1),
                            ):
            if np.mod(np.sum([n, l, m]), 2) == 0:
                Z[n, l, m] = np.real(
                    Z[n, l, m]) - np.imag(Z[n, l, m]) * IMAG_CONST
            else:
                Z[n, l, m] = -np.real(Z[n, l, m]) + \
                    np.imag(Z[n, l, m]) * IMAG_CONST

        return Z

    def Yljm(self, l, j, m):
        aux_1 = np.power(-1, j) * (np.sqrt(2 * l + 1) / np.power(2, l))
        aux_2 = self.trinomial(
            m, j, l - m - 2 * j) * nchoosek(2 * (l - j), l - j)
        aux_3 = np.sqrt(self.trinomial(m, m, l - m))
        y = (aux_1 * aux_2) / aux_3
        return y

    def Qklnu(self, k, l, nu):
        aux_1 = np.power(-1, k + nu) / np.float64(np.power(4, k))
        aux_2 = np.sqrt((2 * l + 4 * k + 3) / 3.0)
        aux_3 = self.trinomial(
            nu, k - nu, l + nu + 1) * nchoosek(2 * (l + nu + 1 + k), l + nu + 1 + k)
        aux_4 = nchoosek(2.0 * (l + nu + 1), l + nu + 1)
        return (aux_1 * aux_2 * aux_3) / aux_4

    def feature_extraction(self, Z, N):
        F = np.zeros([N + 1, N + 1]) - 1  # +NAN_CONST
        for n in range(N + 1):
            for l in range(n + 1):
                if np.mod(n - l, 2) != 0:
                    continue
                aux_1 = Z[n, l, 0:(l + 1)]
                if l > 0:
                    aux_2 = np.conj(aux_1[1:(l + 1)])
                    for m in range(0, l):
                        aux_2[m] = aux_2[m] * np.power(-1, m + 1)
                    aux_2 = np.flipud(aux_2)
                    aux_1 = np.concatenate([aux_2, aux_1])
                F[n, l] = np.linalg.norm(aux_1, ord=2)
        F = F.transpose()
        return F[F >= 0]

class KoehlMultiproc(KoehlOptimizations):
    def geometric_moments_exact(self, points_array, faces_array, N):
        n_facets, n_vertices = faces_array.shape[:2]
        assert n_vertices == 3
        moments_array = np.zeros([N+1, N+1, N+1])
        process_pool = mp.Pool()
        for face in faces_array:
            vertex_list = [points_array[_i-1] for _i in face]
            process_pool.apply_async(_kmp_geometric_moments_exact_worker,
                                     args=(self, vertex_list, N),
                                     callback=moments_array.__iadd__,
                                     )
        process_pool.close()
        process_pool.join()
        return self.factorial_scalar(N) * moments_array


def decimate(points, faces, reduction=0.75, smooth_steps=25,
             scalars=[], save_vtk=False, output_vtk=''):
    # ------------------------------------------------------------------------
    # vtk points:
    # ------------------------------------------------------------------------
    vtk_points = vtk.vtkPoints()
    [vtk_points.InsertPoint(i, x[0], x[1], x[2]) for i,x in enumerate(points)]

    # ------------------------------------------------------------------------
    # vtk faces:
    # ------------------------------------------------------------------------
    vtk_faces = vtk.vtkCellArray()
    for face in faces:
        vtk_face = vtk.vtkPolygon()
        vtk_face.GetPointIds().SetNumberOfIds(3)
        vtk_face.GetPointIds().SetId(0, face[0])
        vtk_face.GetPointIds().SetId(1, face[1])
        vtk_face.GetPointIds().SetId(2, face[2])
        vtk_faces.InsertNextCell(vtk_face)

    # ------------------------------------------------------------------------
    # vtk scalars:
    # ------------------------------------------------------------------------
    if scalars:
        vtk_scalars = vtk.vtkFloatArray()
        vtk_scalars.SetName("scalars")
        for scalar in scalars:
            vtk_scalars.InsertNextValue(scalar)

    # ------------------------------------------------------------------------
    # vtkPolyData:
    # ------------------------------------------------------------------------
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(vtk_points)
    polydata.SetPolys(vtk_faces)
    if scalars:
        polydata.GetPointData().SetScalars(vtk_scalars)

    # ------------------------------------------------------------------------
    # Decimate:
    # ------------------------------------------------------------------------
    # We want to preserve topology (not let any cracks form).
    # This may limit the total reduction possible.
    decimate = vtk.vtkDecimatePro()

    # Migrate to VTK6:
    # http://www.vtk.org/Wiki/VTK/VTK_6_Migration/Replacement_of_SetInput
    # Old: decimate.SetInput(polydata)
    decimate.SetInputData(polydata)

    decimate.SetTargetReduction(reduction)
    decimate.PreserveTopologyOn()

    # ------------------------------------------------------------------------
    # Smooth:
    # ------------------------------------------------------------------------
    if save_vtk:
        if not output_vtk:
            output_vtk = os.path.join(os.getcwd(), 'decimated.vtk')
        exporter = vtk.vtkPolyDataWriter()
    else:
        output_vtk = None
    if smooth_steps > 0:
        smoother = vtk.vtkSmoothPolyDataFilter()

        # Migrate to VTK6:
        # http://www.vtk.org/Wiki/VTK/VTK_6_Migration/Replacement_of_SetInput
        # Old: smoother.SetInput(decimate.GetOutput())
        smoother.SetInputConnection(decimate.GetOutputPort())

        smoother.SetNumberOfIterations(smooth_steps)
        smoother.Update()
        out = smoother.GetOutput()

        # Migrate to VTK6:
        # http://www.vtk.org/Wiki/VTK/VTK_6_Migration/Replacement_of_SetInput
        # Old: exporter.SetInput(smoother.GetOutput())
        exporter.SetInputConnection(smoother.GetOutputPort())

    else:
        decimate.Update()
        out = decimate.GetOutput()
        if save_vtk:
            # Migrate to VTK6:
            # http://www.vtk.org/Wiki/VTK/VTK_6_Migration/Replacement_of_SetInput
            # http://stackoverflow.com/questions/29020740/
            #        what-is-the-difference-in-setinputconnection-and-setinput
            # Old: exporter.SetInput(decimate.GetOutput())
            exporter.SetInputConnection(decimate.GetOutputPort())

    # ------------------------------------------------------------------------
    # Export output:
    # ------------------------------------------------------------------------
    if save_vtk:
        exporter.SetFileName(output_vtk)
        exporter.Write()
        if not os.path.exists(output_vtk):
            raise IOError(output_vtk + " not found")

    # ------------------------------------------------------------------------
    # Extract decimated points, faces, and scalars:
    # ------------------------------------------------------------------------
    points = [list(out.GetPoint(point_id))
              for point_id in range(out.GetNumberOfPoints())]
    if out.GetNumberOfPolys() > 0:
        polys = out.GetPolys()
        pt_data = out.GetPointData()
        faces = [[int(polys.GetData().GetValue(j))
                  for j in range(i*4 + 1, i*4 + 4)]
                  for i in range(polys.GetNumberOfCells())]
        if scalars:
            scalars = [pt_data.GetScalars().GetValue(i)
                       for i in range(len(points))]
    else:
        faces = []
        scalars = []

    return points, faces, scalars, output_vtk

def reindex_faces_0to1(faces):
    faces = [[old_index+1 for old_index in face] for face in faces]

    return faces

DefaultPipeline = type(
    'DefaultPipeline', (KoehlMultiproc, SerialPipeline), {})

def zernike_moments(model, order=10, scale_input=True,
                    decimate_fraction=0, decimate_smooth=0, verbose=False):
    points = model.vertices
    faces = model.faces
    # Convert 0-indices (Python) to 1-indices (Matlab) for all face indices:
    index1 = False  # already done elsewhere in the code
    if index1:
        faces = reindex_faces_0to1(faces)

    # Convert lists to numpy arrays:
    if isinstance(points, list):
        points = np.array(points)
    if isinstance(faces, list):
        faces = np.array(faces)

    # ------------------------------------------------------------------------
    # Translate all points so that they are centered at their mean,
    # and scale them so that they are bounded by a unit sphere:
    # ------------------------------------------------------------------------
    if scale_input:
        center = np.mean(points, axis=0)
        points = points - center
        maxd = np.max(np.sqrt(np.sum(points**2, axis=1)))
        points /= maxd

    # ------------------------------------------------------------------------
    # Decimate surface:
    # ------------------------------------------------------------------------
    if 0 < decimate_fraction < 1:
        points, faces, u1,u2 = decimate(points, faces,
            decimate_fraction, decimate_smooth, [], save_vtk=False)

        # Convert lists to numpy arrays:
        points = np.array(points)
        faces = np.array(faces)

    # ------------------------------------------------------------------------
    # Multiprocessor pipeline:
    # ------------------------------------------------------------------------
    pl = DefaultPipeline()

    # ------------------------------------------------------------------------
    # Geometric moments:
    # ------------------------------------------------------------------------
    G = pl.geometric_moments_exact(points, faces, order)

    # ------------------------------------------------------------------------
    # ------------------------------------------------------------------------
    Z = pl.zernike(G, order)

    # ------------------------------------------------------------------------
    # Extract Zernike descriptors:
    # ------------------------------------------------------------------------
    descriptors = pl.feature_extraction(Z, order).tolist()

    if verbose:
        print("Zernike moments: {0}".format(descriptors))

    return descriptors

def compute_distance_zernike(desc1: np.ndarray, desc2: np.ndarray, metric: str = 'l1') -> float:
    if metric == 'l1':
        return np.sum(np.abs(desc1 - desc2))
    elif metric == 'l2':
        return np.sqrt(np.sum((desc1 - desc2) ** 2))
    else:
        raise ValueError("Unsupported metric. Use 'l1' or 'l2'.")