import os
import numpy as np
import json
from PIL import Image
from model import *
import matplotlib.pyplot as plt
import matplotlib.patches as patches

data_path = '../data/RedLights2011_Medium'
preds_path = '../data/hw01_preds'

kernels = buildKernels()

good_examples = ['RL-011.jpg', 'RL-019.jpg', 'RL-021.jpg', 'RL-031.jpg']
bad_examples = ['RL-046.jpg', 'RL-040.jpg', 'RL-023.jpg', 'RL-017.jpg']


for file_name in good_examples:
    I = Image.open(os.path.join(data_path,file_name))
    fig, ax = plt.subplots()
    ax.imshow(I)
    ax.set_axis_off()

    I = np.asarray(I)
    boxes = detect_red_lights(I, kernels, .95)
    for box in boxes:
        tl_row, tl_col, br_row, br_col = box[0], box[1], box[2], box[3]
        rectangle = patches.Rectangle((tl_col, tl_row), br_col-tl_col, br_row-tl_row, linewidth=1, edgecolor='w', facecolor='none')
        ax.add_patch(rectangle)

    plt.savefig(os.path.join(preds_path,file_name), bbox_inches='tight', pad_inches=0)

for file_name in bad_examples:
    I = Image.open(os.path.join(data_path,file_name))
    fig, ax = plt.subplots()
    ax.imshow(I)
    ax.set_axis_off()

    I = np.asarray(I)
    boxes = detect_red_lights(I, kernels, .95)
    for box in boxes:
        tl_row, tl_col, br_row, br_col = box[0], box[1], box[2], box[3]
        rectangle = patches.Rectangle((tl_col, tl_row), br_col-tl_col, br_row-tl_row, linewidth=1, edgecolor='w', facecolor='none')
        ax.add_patch(rectangle)

    plt.savefig(os.path.join(preds_path,file_name), bbox_inches='tight', pad_inches=0)

