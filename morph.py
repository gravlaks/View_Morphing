from morph_utils import*
import scipy.spatial

# factor
use_prewarp = True
s = 0.5

# load images
I_1, I_2, dim_1 = load_mona_lisas()

# three set of features -- the frame, the first face, the second face
f_0 = np.array([
    [ 0, 0, 1 ],
    [ dim_1[0], 0, 1 ],
    [ 0, dim_1[1], 1 ],
    [ dim_1[0], dim_1[1], 1 ],
])
f_1, f_2 = find_face(I_1), find_face(I_2)

F = get_fundamental(f_1, f_2)

f_1 = np.vstack([ f_1, f_0 ])
f_2 = np.vstack([ f_2, f_0 ])

# find triangulation
delu = scipy.spatial.Delaunay(f_1[:,:2]).simplices

# Find homographies, new image dimension (larger)
H_1, H_2, dim_2 = get_framed_homographies(F, f_1, f_2, dim_1)

Hi_1 = np.linalg.inv(H_1)
Hi_2 = np.linalg.inv(H_2)

# Transform images and features
if use_prewarp:
    I_1 = cv.warpPerspective(I_1, Hi_1, dim_2)
    I_2 = cv.warpPerspective(I_2, Hi_2, dim_2)
    f_1 = apply_perspective(Hi_1, f_1)
    f_2 = apply_perspective(Hi_2, f_2)
else:
    dim_2 = dim_1
f_s = s * f_1 + (1 - s) * f_2

# Find the postwarp transformation
H_s = cv.getPerspectiveTransform(f_s[-4:,:2].astype(np.float32), f_0[:,:2].astype(np.float32))

# Find the coordiantes for all triangles
tset_1 = f_1[delu][:, :, :2]
tset_2 = f_2[delu][:, :, :2]
tset_s = f_s[delu][:, :, :2]

I_s = 0 * I_1.copy()
for t_1, t_2, t_s, t_i in zip(tset_1, tset_2, tset_s, delu):
    if np.any(t_i > 67):
        # do not include background
        continue

    # find the affine transformations which send the triangles in each image to the composed location
    S_1 = cv.getAffineTransform(t_1.astype(np.float32), t_s.astype(np.float32))
    S_2 = cv.getAffineTransform(t_2.astype(np.float32), t_s.astype(np.float32))

    # find the mask for the triangle
    mask = cv.bitwise_and(
        cv.drawContours(0 * I_1.copy(), [t_s.astype(np.int64)], -1, (255, 255, 255), -1),
        cv.bitwise_not(I_s)
    )

    # mix the two images on the correct location
    mix = cv.addWeighted(
        cv.warpAffine(I_1.copy(), S_1, dim_2), s,
        cv.warpAffine(I_2.copy(), S_2, dim_2), 1 - s,
        0
    )

    # add the triangle to the image
    I_s += cv.bitwise_and(mask, mix)

# draw the final image
I_s = cv.warpPerspective(I_s, H_s, dim_1)
cv.imwrite(f'mona{s}.jpg', I_s)