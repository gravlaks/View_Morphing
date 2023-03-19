import cv2
import numpy as np
from morph_utils import*
import scipy.spatial
import utils
from tqdm import tqdm
from itertools import count
import geo_utils
import matplotlib.pyplot as plt
from HW.p4 import *
from HW.p3 import *

def apply_projection(A, x):
    """
        in : 
            A ~ (3, 4) - Projection matrix
            x ~ (N, 4) - homogenous 3d points
        out :
            y ~ (N, 3) - homogenous 2d points
    """
    y = np.einsum('ij,nj->ni', A, x)
    return y / y[:, 2].reshape((-1, 1))

def get_normal_map_from_tris(P, tris):
    """
        P    ~ (N, 4) - 3d points
        tris ~ (M, 3) - triangle indices
    """
    normal_map = dict()
    for tri in tris:
        T = P[tri]
        a = T[0, :3]
        b = T[1, :3]
        c = T[2, :3]
        ijk = tuple(tri)
        ab = (a - b)
        cb = (c - b)
        v = np.cross(ab, cb)
        v /= np.linalg.norm(v)
        normal_map[ijk] = -v
        #normal_map[ijk] *= -np.sign(ref.dot(v))
    return normal_map

def create_RT(pos, dir):
    z = dir / np.linalg.norm(dir)
    RT = np.zeros((3,4))
    R = scipy.spatial.transform.Rotation.from_rotvec(z)
    RT[:3,:3] = R.as_matrix()
    RT[:,3] = - RT[:3, :3] @ pos

    return RT

def get_M(p, P):
    """
        p ~ (N, 3) - 2d homogenous points
        P ~ (N, 4) - 3d homogenous points
    """
    import cvxpy as cp

    X = cp.Variable(4)
    Y = cp.Variable(4)
    Z = cp.Variable(4)
    k = cp.Parameter()

    problem = cp.Problem(cp.Minimize(0), [
        P @ X.T - cp.multiply(p[:, 0], P @ Z.T) <= k * P @ Z.T,
        -k * P @ Z.T <= P @ X.T - cp.multiply(p[:, 0], P @ Z.T),
        P @ Y.T - cp.multiply(p[:, 1], P @ Z.T) <= k * P @ Z.T,
        -k * P @ Z.T <= P @ Y.T - cp.multiply(p[:, 1], P @ Z.T),
    ])

    l = 0
    u = 100000
    e = 0.001
    while u - l > e:
        k.value = 0.5 * (l + u)
        problem.solve()
        if problem.status == 'optimal':
            u = k.value
        else:
            l = k.value
    
    M = np.zeros((3, 4))
    M[0,:] = X.value
    M[1,:] = Y.value
    M[2,:] = Z.value
    M /= M[2,3] * 0.2

    return M

def get_MC(M_1, M_2):
    H_1 = M_1[:3,:3]
    H_2 = M_2[:3,:3]
    C_1 = -np.linalg.solve(H_1, M_1[:, 3])
    C_2 = -np.linalg.solve(H_2, M_2[:, 3])
    return C_1, C_2

def get_KRT(M):
    Q, R = np.linalg.qr(M[:3, :3].T)
    K = R.T
    return K, np.linalg.inv(K) @ M
    KQ = M[:3,:3]
    K = -np.linalg.cholesky((KQ @ KQ.T).T).T
    return K, np.linalg.inv(K) @ M

def get_dir(RT):
    return RT[:3,:3].T @ np.array([ 0, 0, 1 ])

def get_pos(RT):
    return -RT[:3,:3].T @ RT[:,3]

def get_posdir(M):
    k, RT = get_KRT(M)
    return get_pos(RT), 10*get_dir(RT)

def generate_manual(s = 0.5, scale = 1, save = True, subdivide = True):
    # load images
    I_1, f_1, n_1, I_2, f_2, n_2, dim = load_manual(scale)
    f_1 = homogenize_array(f_1)
    f_2 = homogenize_array(f_2)

    # find triangulation
    delu = scipy.spatial.Delaunay(f_1[:,:2]).simplices[:,:]

    # attempt 3d reconstruction of face
    S, _ = factorization_method(f_1, f_2)
    P = np.ones((S.shape[1], 4))
    P[:, :3] = S.T
    M_1 = get_M(f_1, P)
    M_2 = get_M(f_2, P)
    c_1 = np.zeros(3)
    c_2 = np.zeros(3)
    v_1 = np.array([0, +2, 2])
    v_2 = np.array([0, -2, 2])

    #geo_utils.plot_tris_3d_norm(P[:,:3], normal_map, delu)
    #plt.show()
    #c_1, v_1 = get_posdir(M_1)
    #c_2, v_2 = get_posdir(M_2)
    #plt.figure(1)
    #geo_utils.plot_tris_2d(apply_projection(M_1, P)[:,:2], delu)
    #plt.figure(2)
    #geo_utils.plot_tris_2d(apply_projection(M_2, P)[:,:2], delu)
    #plt.figure(0)
    #geo_utils.plot_tris_3d(P[:, :3], delu)
    #plt.quiver(*c_1, *v_1)
    #plt.quiver(*c_2, *v_2)
    #plt.show()

    v_s = v_2 * (1 - s) + v_1 * s

    # find M_s
    M_s = s * M_1 + (1 - s) * M_2

    f_s = apply_projection(M_s, P)
    
    if subdivide:
        for i in range(1):
            Q = geo_utils.catmull_clark(P, delu)
            P = np.ones((Q.shape[0], 4))
            P[:, :3] = Q[:, :3]
            f_1 = apply_projection(M_1, P)
            f_2 = apply_projection(M_2, P)
            f_s = apply_projection(M_s, P)
            delu = scipy.spatial.Delaunay(f_1[:,:2]).simplices

    normal_map = get_normal_map_from_tris(P, delu)

    # Find the coordiantes for all triangles
    tset_1 = f_1[delu][:, :, :2]
    tset_2 = f_2[delu][:, :, :2]
    tset_s = f_s[delu][:, :, :2]

    I_s = 0 * I_1.copy()
    local_mask = 0 * I_1.copy()
    global_mask = 0 * I_1.copy()

    for t_1, t_2, t_s, t_i in zip(tset_1, tset_2, tset_s, delu):
        # find the affine transformations which send the triangles in each image to the composed location
        S_1 = cv.getAffineTransform(t_1.astype(np.float32), t_s.astype(np.float32))
        S_2 = cv.getAffineTransform(t_2.astype(np.float32), t_s.astype(np.float32))

        # find the mask for the triangle
        local_mask = cv.drawContours(0 * local_mask.copy(), [t_s.astype(np.int64)], -1, (255, 255, 255), -1)

        n = normal_map[tuple(t_i)]

        if n.dot(v_s) < 0:
            continue

        s_1 = (0.01 - min(n.dot(v_1), 0))
        s_2 = (0.01 - min(n.dot(v_2), 0))
        S = s_1 + s_2
        s_1 = s_1 / S
        s_2 = s_2 / S

        mix = cv.addWeighted(
            cv.warpAffine(I_1.copy(), S_1, dim), s_1,
            cv.warpAffine(I_2.copy(), S_2, dim), s_2,
            0
        )

        I_s |= local_mask & mix & ~global_mask
        global_mask |= local_mask

    if save:
        cv.imwrite(f'output/occlusion_{s:.2f}.jpg', I_s)

    return I_s.copy()

if __name__ == "__main__":
    #generate_manual(s=0.5, scale=0.4, save=True, subdivide=True)
    I = []
    for s in np.linspace(0, 1, 7):
        I += [ generate_manual(s=s, scale=0.4, save=False, subdivide=True) ]
    utils.create_gif('output/occlusion.gif', I + I[::-1])