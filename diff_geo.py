import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from scipy import spatial
import cvxpy as cp
from itertools import permutations

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
    for i, j, k in tris:
        tri = np.array([i, j, k, i])
        trip = pts_3d[tri]
        ax.plot(*trip.T)

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

# 3d points ~ (N, 3)
pts_3d = np.vstack([X, Y, Z]).T

# 2d points (projection onto xy) ~ (N, 2)
pts_2d = pts_3d[:, :2]

# find triangulation of projection
tris = spatial.Delaunay(pts_2d).simplices

def get_tri_indices(pts_3d, tris):
    """
        Return every tri_col[i] is every triangle connected to the index i
    """
    indices = np.arange(pts_3d.shape[0])
    tri_col = list()

    for index in indices:
        col = list()
        for i, j, k in tris:
            if index == i: col += [ np.array([ k, i, j]) ]
            if index == j: col += [ np.array([ i, j, k]) ]
            if index == k: col += [ np.array([ j, k, i]) ]
        tri_col += [ col ]

    return indices, tri_col

def get_angle_area(ftri):
    """
        Return the angle and area of the 3d triangle with vertices ftri = [ v1, v2, v3 ] (angle is between v1 and v3 seen from v2)
    """
    I, J, K = ftri[0], ftri[1], ftri[2]
    V = I - J
    W = K - J
    V_l = np.linalg.norm(V)
    W_l = np.linalg.norm(W)
    VcW = np.linalg.norm(np.cross(V, W))
    return np.arcsin(VcW / (V_l * W_l)), VcW / 2

def get_tri_curvature(pts_3d, tris):
    """
        Calculate the curvature of every index
    """
    indices, tri_col = get_tri_indices(pts_3d, tris)
    curvatures = []

    for index in indices:
        angle_sum = 0
        area_sum = 0

        for tri in tri_col[index]:
            angle, area = get_angle_area(pts_3d[tri])
            angle_sum += angle
            area_sum += area

        curvatures += [ 3 * (2 * np.pi - angle_sum) / area_sum ]
    
    return indices, curvatures

def get_variable_constraint_pairs(pts_3d, tris):
    """
        Subdivide every triangle into three new triangles.
        For every new triangle add two angles the area as cvx variables.
        Return all the variables as well as constraints which preserve the curvature of all existing points
    """
    indices, curvatures = get_tri_curvature(pts_3d, tris)

    K0 = tris.shape[0] # how many new triangles to add
    K = K0

    tri_dec = [ [] for _ in indices ]

    new_angle_l = dict()
    new_angle_r = dict()
    new_area = dict()
    variables = dict()

    curvature_constraints = []
    sign_constraints = []

    s = lambda x: tuple(sorted(x))

    for i, j, k in tris:
        a, b, c = (i, j, K), (i, K, k), (K, j, k)
        A, B, C = (k, j, i)

        tri_dec[i] += [ a, b ]
        tri_dec[j] += [ a, c ]
        tri_dec[k] += [ b, c ]

        variables[(i, j, k)] = dict()
        variables[(i, j, k)][a] = dict()
        variables[(i, j, k)][b] = dict()
        variables[(i, j, k)][c] = dict()

        for h, v in zip([A, B, C], [a, b, c]):
            new_angle_l[v] = cp.Variable()
            new_angle_r[v] = cp.Variable()
            new_area[v] = cp.Variable()

            variables[(i, j, k)][v]['l'] = new_angle_l[v]
            variables[(i, j, k)][v]['r'] = new_angle_r[v]
            variables[(i, j, k)][v]['t'] = new_area[v]
            variables[(i, j, k)][v]['h'] = pts_3d[h]

        K += 1

    for v in new_angle_l:
        sign_constraints += [
            new_angle_l[v] >= 0,
            new_angle_r[v] >= 0,
            new_area[v] >= 0,
        ]
    
    for index in indices:
        angle_sum = 0
        area_sum = 0
        for ijk in tri_dec[index]:
            angle_sum += new_angle_l[ijk] + new_angle_r[ijk]
            area_sum += new_area[ijk]
        
        curvature_constraints += [ curvatures[i] * area_sum + angle_sum == 2 * np.pi ]
    
    return variables, sign_constraints, curvature_constraints

# Get variables and constraints
variables, sign_constraints, curvature_constraints = get_variable_constraint_pairs(pts_3d, tris)
constraints = sign_constraints + curvature_constraints

# Set up and solve the problem
problem = cp.Problem(cp.Minimize(cp.norm_inf(cp.vstack(variables))), constraints)
problem.solve()

# Figure out where all the new vertices are
for ijk in tris:
    group = variables[tuple(ijk)]
    a, b, c = group
    angle_a_l, angle_a_r, area_a, D_a = group[a]['l'].value, group[a]['r'].value, group[a]['t'].value, group[a]['h']
    angle_b_l, angle_b_r, area_b, D_b = group[b]['l'].value, group[b]['r'].value, group[b]['t'].value, group[b]['h']
    angle_c_l, angle_c_r, area_c, D_c = group[c]['l'].value, group[c]['r'].value, group[c]['t'].value, group[c]['h']

    # TODO: this

#plot_3d(pts_3d)
#plot_tris_2d(pts_2d, tris)
#plot_tris_3d(pts_3d, tris)
#plt.show()
