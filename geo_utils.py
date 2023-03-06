import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from scipy import spatial
from itertools import permutations, chain

figure_count = 0

def get_unique_fig():
    """
        Simply get the handle to a unique matplotlib window
    """
    global figure_count
    fig = plt.figure(figure_count)
    figure_count += 1
    return fig

def plot_3d(pts_3d):
    """
        Plot points in 3d
    """
    fig = get_unique_fig()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.plot(*pts_3d.T, 'o')

def plot_tris_2d(pts_2d, tris):
    """
        Plot a triangle mesh in 2d
    """
    fig = get_unique_fig()
    ax = fig.gca()
    for i, j, k in tris:
        tri = np.array([i, j, k, i])
        trip = pts_2d[tri]
        ax.plot(*trip.T)

def plot_tris_3d(pts_3d, tris):
    """
        Plot a triangle mesh in 3d
    """
    fig = get_unique_fig()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    x_min = np.min(pts_3d[:,:])
    x_max = np.max(pts_3d[:,:])
    for i, j, k in tris:
        tri = np.array([i, j, k, i])
        trip = pts_3d[tri]
        ax.plot(*trip.T)

# find triangulation of projection
def get_tris_xy(pts_3d):
    """
        Project onto xy-plane and find triangulation
    """
    return spatial.Delaunay(pts_3d[:, :2]).simplices

def get_edges_from_triangle(i, j, k):
    """
        (i, j, k) -> [(i, j), (j, k), (k, i)]
        (sorts the edges so that adjacent triangles generate the same common edge)
    """
    return list(map(tuple, map(sorted, [ (i, j), (j, k), (k, i) ])))

def edges_from_tris(tris):
    """
        Take every triangle in triangle mesh and generate every unique edge
    """
    return list(set(chain(*[get_edges_from_triangle(i, j, k) for i, j, k in tris])))

def edge_ad_index(pts_3d, edges, tris):
    """
        i -> ?e1, ?e2, ...
        Get every edge adjacent to every vertex (by index)
    """
    indices = np.arange(pts_3d.shape[0])

    mapping = { index : [] for index in indices }

    for (i, j) in edges:
        mapping[i] += [ (i, j) ]
        mapping[j] += [ (i, j) ]
    
    return mapping

def tri_ad_index(pts_3d, tris):
    """
        i -> ?T1, ?T2, ...
        Get every triangle adjacent to every vertex (by index)
    """
    indices = np.arange(pts_3d.shape[0])
    mapping = { index : [] for index in indices }

    for i, j, k in tris:
        mapping[i] += [ (i, j, k) ]
        mapping[j] += [ (i, j, k) ]
        mapping[k] += [ (i, j, k) ]

    return mapping

def tri_ad_edge(edges, tris):
    """
        e -> ?T1, ?T2, ...
        Get every triangle adjacent to every edge
    """
    mapping = { edge : [] for edge in edges }

    for (i, j, k) in tris:
        e1, e2, e3 = get_edges_from_triangle(i, j, k)
        mapping[e1] += [ (i, j, k) ]
        mapping[e2] += [ (i, j, k) ]
        mapping[e3] += [ (i, j, k) ]
    
    return mapping

def vface_tri(pts_3d, tris):
    """
        T -> v_T
        Get the midpoint of a triangle
    """
    mapping = { (i, j, k) : 0 for (i, j, k) in tris }
    for tri in tris:
        ijk = tuple(tri)
        mapping[ijk] = np.mean(pts_3d[tri], axis = 0) # check

    return mapping

def vwedge_edge(pts_3d, edges, tris):
    """
        e -> e'
        Get the new edge point for every edge
    """
    TAE = tri_ad_edge(edges, tris)
    VFT = vface_tri(pts_3d, tris)

    mapping = dict()

    for i, j in edges:
        k = 2
        T = 0
        for t in TAE[(i, j)]:
            T += VFT[t]
            k += 1
        v, w = pts_3d[i], pts_3d[j]
        mapping[(i, j)] = (T + v + w) / k
    
    return mapping

def vmedge_edge(pts_3d, edges, tris):
    """
        (i, j) -> (v_i + v_j) / 2
        Get the midpoint of every edge
    """
    mapping = dict()

    for i, j in edges:
        v, w = pts_3d[i], pts_3d[j]
        mapping[(i, j)] = (v + w) / 2
    
    return mapping

def catmull_clark(pts_3d, tris):
    """
        Catmull-clark subdivision (without mesh generation)
        Source : https://web.cse.ohio-state.edu/~dey.8/course/784/note20.pdf
        Takes:
         - pts_3d ~ (N, 3)
         - tris ~ (M, 3)
        Returns:
         - pts_3d ~ (K, 3)
    """
    edges = edges_from_tris(tris)
    v_edge_mid = vmedge_edge(pts_3d, edges, tris)
    v_edge_new = vwedge_edge(pts_3d, edges, tris)
    v_face_new = vface_tri(pts_3d, tris)

    EAI = edge_ad_index(pts_3d, edges, tris)
    TAE = tri_ad_edge(edges, tris)
    TAI = tri_ad_index(pts_3d, tris)

    points = []
    points += [ v_face_new[tuple(tri)] for tri in tris ]
    points += [ v_edge_new[edge] for edge in edges ]
    for i, pt in enumerate(pts_3d):
        n = len(EAI[i])
        R = np.mean(np.array([ v_edge_mid[edge] for edge in EAI[i] ]), axis = 0)
        Q = np.mean(np.array([ v_face_new[tri] for tri in TAI[i] ]), axis = 0)
        points += [ 1 / n * Q + 2 / n * R + (n - 3) / n * pt]
    
    return np.array(points)

if __name__ == '__main__':
    # generate sphere octant
    N = 5
    u = np.linspace(0.25 * np.pi, 0.50 * np.pi, N)
    v = np.linspace(0.12 * np.pi, 0.36 * np.pi, N)
    u, v = np.meshgrid(u, v)
    u = u.ravel()
    v = v.ravel()
    ones = np.ones(N)
    X = np.sin(u) * np.sin(v)
    Y = np.cos(u) * np.sin(v)
    Z = np.cos(v)
    
    # Compose sphere octant, find deluaney triangulation, plot
    pts_3d = np.vstack([X, Y, Z]).T
    tris = get_tris_xy(pts_3d)
    plot_tris_3d(pts_3d, tris)
    plt.title("Original mesh")
    
    # Generate new points, find deluaney triangulation, plot
    pts_3d = catmull_clark(pts_3d, tris)
    tris = get_tris_xy(pts_3d)
    plot_tris_3d(pts_3d, tris)
    plt.title("Subdivided mesh")
    
    plt.show()
    