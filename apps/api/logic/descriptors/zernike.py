import numpy as np
from scipy.special import factorial, comb

IMAG_CONST = 1j
PI_CONST = np.pi


def nested_loop(stack, args):
    if len(stack) != 0:
        fn = stack.pop()
        for i in fn(*args):
            for j in nested_loop(stack, args + [i]):
                yield (i,) + j
        stack.append(fn)
    else:
        yield tuple()


def nest(*_stack):
    return nested_loop(list(reversed(_stack)), [])


def autocat(arrs, axis=None, **dargs):
    if axis is None:
        return np.concatenate(arrs, **dargs)
    ndim = arrs[0].ndim
    assert all([a.ndim == ndim for a in arrs])
    if axis >= ndim:
        arrs = tuple([np.expand_dims(a, axis) for a in arrs])
    return np.concatenate(arrs, axis=axis)


class MomentsCalculator:
    def geometric_moments_exact(self, points_array, faces_array, N):
        n_facets, n_vertices = faces_array.shape[:2]
        assert n_vertices == 3
        moments_array = np.zeros([N + 1, N + 1, N + 1])
        monomial_array = self.monomial_precalc(points_array, N)
        for face in faces_array:
            vertex_list = [points_array[_i - 1, ...] for _i in face]
            Cf_list = [monomial_array[_i - 1, ...] for _i in face]
            Vf = self.facet_volume(vertex_list)
            moments_array += Vf * self.term_Sijk(Cf_list, N)
        return self.factorial_scalar(N) * moments_array

    def factorial_scalar(self, N):
        i, j, k = np.mgrid[0 : N + 1, 0 : N + 1, 0 : N + 1]
        return (
            factorial(i)
            * factorial(j)
            * factorial(k)
            / (factorial(i + j + k + 2) * (i + j + k + 3))
        )

    def monomial_precalc(self, points_array, N):
        n_points = points_array.shape[0]
        monomial_array = np.zeros([n_points, N + 1, N + 1, N + 1])
        tri_array = self.trinomial_precalc(N)
        for point_indx, point in enumerate(points_array):
            monomial_array[point_indx, ...] = self.mon_comb(point, tri_array, N)
        return monomial_array

    def mon_comb(self, vertex, tri_array, N):
        x, y, z = vertex
        c = np.zeros([N + 1, N + 1, N + 1])
        for i, j, k in nest(
            lambda: range(N + 1),
            lambda _i: range(N - _i + 1),
            lambda _i, _j: range(N - _i - _j + 1),
        ):
            c[i, j, k] = (
                tri_array[i, j, k] * np.power(x, i) * np.power(y, j) * np.power(z, k)
            )
        return c

    def trinomial_precalc(self, N):
        tri_array = np.zeros([N + 1, N + 1, N + 1])
        for i, j, k in nest(
            lambda: range(N + 1),
            lambda _i: range(N - _i + 1),
            lambda _i, _j: range(N - _i - _j + 1),
        ):
            tri_array[i, j, k] = self.trinomial(i, j, k)
        return tri_array

    def trinomial(self, i, j, k):
        return factorial(i + j + k) / (factorial(i) * factorial(j) * factorial(k))

    def facet_volume(self, vertex_list):
        return np.linalg.det(autocat(vertex_list, axis=1))

    def term_Sijk(self, Cf_list, N):
        S = np.zeros([N + 1, N + 1, N + 1])
        C0, C1, C2 = Cf_list
        Dabc = self.term_Dabc(C1, C2, N)
        for i, j, k, ii, jj, kk in nest(
            lambda: range(N + 1),
            lambda _i: range(N - _i + 1),
            lambda _i, _j: range(N - _i - _j + 1),
            lambda _i, _j, _k: range(_i + 1),
            lambda _i, _j, _k, _ii: range(_j + 1),
            lambda _i, _j, _k, _ii, _jj: range(_k + 1),
        ):
            S[i, j, k] += C0[ii, jj, kk] * Dabc[i - ii, j - jj, k - kk]
        return S

    def term_Dabc(self, C1, C2, N):
        D = np.zeros([N + 1, N + 1, N + 1])
        for i, j, k, ii, jj, kk in nest(
            lambda: range(N + 1),
            lambda _i: range(N + 1),
            lambda _i, _j: range(N + 1),
            lambda _i, _j, _k: range(_i + 1),
            lambda _i, _j, _k, _ii: range(_j + 1),
            lambda _i, _j, _k, _ii, _jj: range(_k + 1),
        ):
            D[i, j, k] += C1[ii, jj, kk] * C2[i - ii, j - jj, k - kk]
        return D

    def zernike(self, G, N):
        V = np.zeros([N + 1, N + 1, N + 1], dtype=complex)
        for a, b, c, alpha in nest(
            lambda: range(int(N / 2) + 1),
            lambda _a: range(N - 2 * _a + 1),
            lambda _a, _b: range(N - 2 * _a - _b + 1),
            lambda _a, _b, _c: range(_a + _c + 1),
        ):
            V[a, b, c] += (
                np.power(IMAG_CONST, alpha)
                * comb(a + c, alpha)
                * G[2 * a + c - alpha, alpha, b]
            )

        W = np.zeros([N + 1, N + 1, N + 1], dtype=complex)
        for a, b, c, alpha in nest(
            lambda: range(int(N / 2) + 1),
            lambda _a: range(N - 2 * _a + 1),
            lambda _a, _b: range(N - 2 * _a - _b + 1),
            lambda _a, _b, _c: range(_a + 1),
        ):
            W[a, b, c] += (
                np.power(-1, alpha)
                * np.power(2, a - alpha)
                * comb(a, alpha)
                * V[a - alpha, b, c + 2 * alpha]
            )

        X = np.zeros([N + 1, N + 1, N + 1], dtype=complex)
        for a, b, c, alpha in nest(
            lambda: range(int(N / 2) + 1),
            lambda _a: range(N - 2 * _a + 1),
            lambda _a, _b: range(N - 2 * _a - _b + 1),
            lambda _a, _b, _c: range(_a + 1),
        ):
            X[a, b, c] += comb(a, alpha) * W[a - alpha, b + 2 * alpha, c]

        Y = np.zeros([N + 1, N + 1, N + 1], dtype=complex)
        for l, nu, m, j in nest(
            lambda: range(N + 1),
            lambda _l: range(int((N - _l) / 2) + 1),
            lambda _l, _nu: range(_l + 1),
            lambda _l, _nu, _m: range(int((_l - _m) / 2) + 1),
        ):
            Y[l, nu, m] += self.Yljm(l, j, m) * X[nu + j, l - m - 2 * j, m]

        Z = np.zeros([N + 1, N + 1, N + 1], dtype=complex)
        for n, l, m, nu in nest(
            lambda: range(N + 1),
            lambda _n: range(_n + 1),
            lambda _n, _l: range(_l + 1),
            lambda _n, _l, _m: range(int((_n - _l) / 2) + 1),
        ):
            k = int((n - l) / 2)
            Z[n, l, m] += (
                (3 / (4 * PI_CONST)) * self.Qklnu(k, l, nu) * np.conj(Y[l, nu, m])
            )

        for n, l, m in nest(
            lambda: range(N + 1), lambda _n: range(n + 1), lambda _n, _l: range(l + 1)
        ):
            if np.mod(np.sum([n, l, m]), 2) == 0:
                Z[n, l, m] = np.real(Z[n, l, m]) - np.imag(Z[n, l, m]) * IMAG_CONST
            else:
                Z[n, l, m] = -np.real(Z[n, l, m]) + np.imag(Z[n, l, m]) * IMAG_CONST

        return Z

    def Yljm(self, l, j, m):
        aux_1 = np.power(-1, j) * (np.sqrt(2 * l + 1) / np.power(2, l))
        aux_2 = self.trinomial(m, j, l - m - 2 * j) * comb(2 * (l - j), l - j)
        aux_3 = np.sqrt(self.trinomial(m, m, l - m))
        return (aux_1 * aux_2) / aux_3

    def Qklnu(self, k, l, nu):
        aux_1 = np.power(-1, k + nu) / np.float64(np.power(4, k))
        aux_2 = np.sqrt((2 * l + 4 * k + 3) / 3.0)
        aux_3 = self.trinomial(nu, k - nu, l + nu + 1) * comb(
            2 * (l + nu + 1 + k), l + nu + 1 + k
        )
        aux_4 = comb(2.0 * (l + nu + 1), l + nu + 1)
        return (aux_1 * aux_2 * aux_3) / aux_4

    def feature_extraction(self, Z, N):
        F = np.zeros([N + 1, N + 1]) - 1
        for n in range(N + 1):
            for l in range(n + 1):
                if np.mod(n - l, 2) != 0:
                    continue
                aux_1 = Z[n, l, 0 : (l + 1)]
                if l > 0:
                    aux_2 = np.conj(aux_1[1 : (l + 1)])
                    for m in range(0, l):
                        aux_2[m] = aux_2[m] * np.power(-1, m + 1)
                    aux_2 = np.flipud(aux_2)
                    aux_1 = np.concatenate([aux_2, aux_1])
                F[n, l] = np.linalg.norm(aux_1, ord=2)
        F = F.transpose()
        return F[F >= 0]


def zernike_moments(model, order=3, scale_input=True):
    points = model.vertices
    faces = model.faces
    if isinstance(points, list):
        points = np.array(points)
    if isinstance(faces, list):
        faces = np.array(faces)

    if scale_input:
        center = np.mean(points, axis=0)
        points = points - center
        maxd = np.max(np.sqrt(np.sum(points**2, axis=1)))
        points /= maxd

    calculator = MomentsCalculator()
    G = calculator.geometric_moments_exact(points, faces, order)
    Z = calculator.zernike(G, order)
    descriptors = calculator.feature_extraction(Z, order).tolist()

    return descriptors


def compute_distance_zernike(
    desc1: np.ndarray, desc2: np.ndarray, metric: str = "l1"
) -> float:
    if metric == "l1":
        return np.sum(np.abs(desc1 - desc2))
    elif metric == "l2":
        return np.sqrt(np.sum((desc1 - desc2) ** 2))
    else:
        raise ValueError("Unsupported metric. Use 'l1' or 'l2'.")
