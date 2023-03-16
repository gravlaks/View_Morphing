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
        a, b, c = np.split(P[tri], [1, 2])
        a = a[0, :3]
        b = b[0, :3]
        c = c[0, :3]
        ijk = tuple(tri)
        v = np.cross(a-b, c-b)
        v /= np.linalg.norm(v)
        normal_map[ijk] = v
    return normal_map


def generate_manual(s = 0.5, scale = 1, save = True, subdivide = True):
    # load images
    I_1, f_1, n_1, I_2, f_2, n_2, dim = load_manual(scale)
    f_1 = homogenize_array(f_1)
    f_2 = homogenize_array(f_2)
    #f_1 = find_face(I_1)
    #f_2 = find_face(I_2)

    f_count = len(f_1)


    # find triangulation
    delu = scipy.spatial.Delaunay(f_1[:,:2]).simplices

    F = np.zeros((f_count, 2, 2))
    F[:, 0, :] = f_1[:,:2]
    F[:, 1, :] = f_2[:,:2]

    # attempt 3d reconstruction of face
    E, K = get_calibration(f_1, f_2)
    RT = estimate_RT_from_E(E, F, K)

    epilines_1 = np.array([E @ f_2[0], E @ f_2[1]])
    epilines_2 = np.array([E.T @ f_1[0], E.T @ f_1[1]])

    v_1 = np.cross(epilines_1[0], epilines_1[1])
    v_2 = np.cross(epilines_2[0], epilines_2[1])
    v_1 = v_1 / np.linalg.norm(v_1)
    v_2 = v_2 / np.linalg.norm(v_2)


    M_1 = K @ np.array([
        [ 1, 0, 0, 0, ],
        [ 0, 1, 0, 0, ],
        [ 0, 0, 1, 0, ],
    ])

    M_2 = K @ RT

    M = np.zeros((2, 3, 4))
    M[0, :, :] = M_1
    M[1, :, :] = M_2

    P = np.ones((f_count, 4))
    for k in range(f_count):
        P[k,:3] = nonlinear_estimate_3d_point(F[k], M)

    p_1 = apply_projection(M_1, P)[:, :2]
    p_2 = apply_projection(M_2, P)[:, :2]
    M_s = s * M_1 + (1 - s) * M_2
    f_s = apply_projection(M_s, P)[:, :2]

    show_faces = False
    if show_faces == True:
        geo_utils.plot_tris_2d(p_1, delu)
        plt.show()

        geo_utils.plot_tris_2d(p_2, delu)
        plt.show()

        geo_utils.plot_tris_2d(f_s, delu)
        plt.show()
    
    show_triang = False

    if show_triang:
        geo_utils.plot_tris_3d(P[:,:3], delu)
        plt.show()
    
    if subdivide:
        for i in range(1):
            Q = geo_utils.catmull_clark(P, delu)
            P = np.ones((Q.shape[0], 4))
            P[:, :3] = Q[:, :3]
            f_1 = apply_projection(M_1, P)
            f_2 = apply_projection(M_2, P)
            f_s = apply_projection(M_s, P)
            delu = scipy.spatial.Delaunay(f_1[:,:2]).simplices

        if show_triang:
            geo_utils.plot_tris_3d(P[:,:3], delu)
            plt.show()

    normal_map = get_normal_map_from_tris(P, delu)

    # Find the coordiantes for all triangles
    tset_1 = f_1[delu][:, :, :2]
    tset_2 = f_2[delu][:, :, :2]
    tset_s = f_s[delu][:, :, :2]

    I_s = 0 * I_1.copy()
    local_mask = 0 * I_1.copy()

    for t_1, t_2, t_s, t_i in zip(tset_1, tset_2, tset_s, delu):
        # find the affine transformations which send the triangles in each image to the composed location
        S_1 = cv.getAffineTransform(t_1.astype(np.float32), t_s.astype(np.float32))
        S_2 = cv.getAffineTransform(t_2.astype(np.float32), t_s.astype(np.float32))

        # find the mask for the triangle
        local_mask = cv.drawContours(0 * local_mask.copy(), [t_s.astype(np.int64)], -1, (255, 255, 255), -1)

        if normal_map[tuple(t_i)] @ v_2 <= 0.5: #then triangle can't be seen from camera 1
            #print(normal_map[tuple(t_i)] @ v_2)
            #mix = cv.warpAffine(I_2.copy(), S_2, dim)
            mix = cv.addWeighted(
                cv.warpAffine(I_1.copy(), S_1, dim), 0.1,
                cv.warpAffine(I_2.copy(), S_2, dim), 0.9,
                0
            )
        elif normal_map[tuple(t_i)] @ v_1 <= 0.5:
            #print("hello")
            #mix = cv.warpAffine(I_1.copy(), S_1, dim)
            mix = cv.addWeighted(
                cv.warpAffine(I_1.copy(), S_1, dim), 0.9,
                cv.warpAffine(I_2.copy(), S_2, dim), 0.1,
                0
            )
        else:
        # mix the two images on the correct location
            mix = cv.addWeighted(
                cv.warpAffine(I_1.copy(), S_1, dim), s,
                cv.warpAffine(I_2.copy(), S_2, dim), 1 - s,
                0
            )
        '''
        mix = cv.addWeighted(
            cv.warpAffine(I_1.copy(), S_1, dim), s,
            cv.warpAffine(I_2.copy(), S_2, dim), 1 - s,
            0
        )
        '''
        I_s |= local_mask & mix
    
    if save:
        cv.imwrite(f'output/occlusion_0.5_newmix3thresh_{s:.2f}.jpg', I_s)

    return I_s.copy()

if __name__ == "__main__":
    I = generate_manual(s=0.5, scale=0.4, save=True, subdivide=True)
    rgb = cv.cvtColor(I, cv.COLOR_BGR2RGB)
    plt.imshow(rgb)
    plt.show()