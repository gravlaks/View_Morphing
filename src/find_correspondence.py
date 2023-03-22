from morph_utils import*
import scipy.spatial

# This is a utility meant to make the evaluation of feature detection easier

I_1, f_1, n_1, I_2, f_2, n_2, dims = load_manual()
f_1 = homogenize_array(f_1)
f_2 = homogenize_array(f_2)

# find triangulation
delu = scipy.spatial.Delaunay(f_1[:,:2]).simplices

for x, y, z in f_1:
    X = x.astype(np.int64)
    Y = y.astype(np.int64)
    I_1 = cv.circle(I_1, (X, Y), 6, (255, 0, 0), -1)

for t in f_1[delu][:, :, :2]:
    I_1 = cv.drawContours(I_1, [t.astype(np.int64)], -1, (255, 255, 255), 2)

for x, y, z in f_2:
    X = x.astype(np.int64)
    Y = y.astype(np.int64)
    I_2 = cv.circle(I_2, (X, Y), 6, (255, 0, 0), -1)

for t in f_2[delu][:, :, :2]:
    I_2 = cv.drawContours(I_2, [t.astype(np.int64)], -1, (255, 255, 255), 2)

fig, (ax1, ax3, ax2) = plt.subplots(1, 3)
ax1.imshow(I_1)
ax2.imshow(I_2)
plt.show()
