import torch
import torch.nn as nn
import numpy as np 

def convert_lbl2color(img):
    # convert label to color
    label_to_color = {
        0: [128, 64,128],
        1: [244, 35,232],
        2: [ 70, 70, 70],
        3: [102,102,156],
        4: [190,153,153],
        5: [153,153,153],
        6: [250,170, 30],
        7: [220,220,  0],
        8: [107,142, 35],
        9: [152,251,152],
        10: [ 70,130,180],
        11: [220, 20, 60],
        12: [255,  0,  0],
        13: [  0,  0,142],
        14: [  0,  0, 70],
        15: [  0, 60,100],
        16: [  0, 80,100],
        17: [  0,  0,230],
        18: [119, 11, 32],
        19: [81,  0, 81]
    }
    # get the image shape
    img_h, img_w = img.shape
    # construct output image from the labels
    output_img = np.zeros((img_h, img_w, 3))
    for row in range(img_h):
        for col in range(img_w):
            label = img[row, col]
            output_img[row, col] = np.array(label_to_color[label])
    
    return output_img