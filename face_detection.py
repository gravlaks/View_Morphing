import dlib
import matplotlib.pyplot as plt
from imutils import face_utils
from HW import p2
from utils import *
from PIL import Image, ImageDraw
import face_recognition_models

if __name__ == '__main__':
    cnn_face_detection_model = face_recognition_models.cnn_face_detector_model_location()
    detector = dlib.cnn_face_detection_model_v1(cnn_face_detection_model)
    predictor = dlib.shape_predictor("data/shape_predictor_68_face_landmarks.dat")

    filename = ""
    im  = cv.imread("output/occlusion_0.5_newmix3thresh_0.50.jpg")
    dims = im.shape
    scale_percent = 30 # percent of original size
    width = int(dims[0] * scale_percent / 100)
    height = int(dims[1]* scale_percent / 100)
    dims = (width, height)
    print()
    im = cv.resize(im, dims, interpolation = cv.INTER_AREA)
    cv.imwrite("temp_data/new_img.png", im)
    dims = im.shape
    mask = np.zeros(dims)
    print(im.shape)
    rects = detector(im, 1)
    print(len(rects))

    
    print(rects)
    for rect in rects:  

        #masks.append(np.zeros_like(np.asarray(im)[:, :, 0]))
        landmarks = predictor(im, rect.rect)
        
        pts = face_utils.shape_to_np(landmarks)
        pts = np.clip(pts, 0, dims[0]-1)
        print(pts.shape)
        convexhull = cv.convexHull(pts).reshape((-1, 2))
        #masks[-1][convexhull[:, 1], convexhull[:, 0]] = 255

        cv.fillPoly(mask, pts=[pts], color=(255, 0, 0))
        cv.imshow("im2", mask)
        cv.waitKey(0)

    np.save("temp_data/mask", mask[:, :, 0])
    
        
