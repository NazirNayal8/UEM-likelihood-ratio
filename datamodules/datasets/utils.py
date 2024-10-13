import numpy as np
from PIL import Image

def read_image(path):
    
    img = np.array(Image.open(path).convert('RGB'))
        
    return img

def read_label(path):
    
    lbl = np.array(Image.open(path))
    
    return lbl