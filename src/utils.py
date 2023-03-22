import numpy as np
from PIL import Image as im
import cv2 as cv

def create_gif(name, frames):
    frames = [ (frame).astype(np.uint8) for frame in frames ]
    frames = [ im.fromarray(frame[:,:,[2, 1, 0]]) for frame in frames ]
    frames[0].save(fp=name, format='GIF', append_images=frames[1:], save_all=True, duration=1, loop=0)

def drawlines(img1,img2,lines,pts1,pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r,c = img1.shape
    img1 = cv.cvtColor(img1,cv.COLOR_GRAY2BGR)
    img2 = cv.cvtColor(img2,cv.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv.line(img1, (x0,y0), (x1,y1), color,1)
        pt1 = tuple([int(pt1[0]), int(pt1[1])])
        pt2 = tuple([int(pt2[0]), int(pt2[1])])
        img1 = cv.circle(img1,pt1,5,color,-1)
        img2 = cv.circle(img2,pt2,5,color,-1)
    return img1,img2
