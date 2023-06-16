import numpy as np
from skimage.transform import resize
from skimage.color import rgb2gray
from config import *

def find_max_lives(env):
    env.reset()
    _, _, _, info = env.step(0)
    return info['lives']

def check_live(life, cur_life):
    if life > cur_life:
        return True
    else:
        return False

def get_frame(X):
    if isinstance(X, tuple):
        X = X[0]  # Assuming the RGB frame is the first element in the tuple
    x = np.uint8(resize(rgb2gray(X), (HEIGHT, WIDTH), mode='reflect') * 255)
    return x



def get_init_state(history, s):
    for i in range(HISTORY_SIZE):
        history[i, :, :] = get_frame(s)
        
def get_init_state(history, s):
    if isinstance(s, tuple):
        s = s[0]  # Extract the image data from the state tuple
    for i in range(HISTORY_SIZE):
        history[i, :, :] = get_frame(s)
