import numpy as np
from PIL import Image as im

def create_gif(name, frames):
    frames = [ (frame).astype(np.uint8) for frame in frames ]
    frames = [ im.fromarray(frame[:,:,[2, 1, 0]]) for frame in frames ]
    frames[0].save(fp=name, format='GIF', append_images=frames[1:], save_all=True, duration=1, loop=0)

