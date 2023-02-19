from morph_utils import*
import scipy.spatial

# This is a utility meant to make the evaluation of feature detection easier

I_1, I_2, dims = load_mona_lisas()
f_1, f_2 = find_face(I_1), find_face(I_2)

f_0 = np.array([
    [ 0, 0, 1 ],
    [ dims[0], 0, 1 ],
    [ 0, dims[1], 1 ],
    [ dims[0], dims[1], 1 ],
])

for x, y, z in np.vstack([f_0, f_1])[::20]:
    X = x.astype(np.int64)
    Y = y.astype(np.int64)
    I_1 = cv.circle(I_1, (X, Y), 3, (255, 0, 0), 1)

for x, y, z in np.vstack([f_0, f_2])[::20]:
    X = x.astype(np.int64)
    Y = y.astype(np.int64)
    I_2 = cv.circle(I_2, (X, Y), 3, (255, 0, 0), 1)

cv.imwrite('output/1.jpg', I_1)
cv.imwrite('output/2.jpg', I_2)