import os
from PIL import Image
import numpy as np
import json
import multiprocessing as mp

def augmentedKernels(kernels):
    scale_list = [1.5, 2., 3.0]
    kernels_out = []
    for kernel in kernels:
        kernels_out.append(kernel)
        kernel = Image.fromarray(kernel)
        w, h = kernel.size
        for scale in scale_list:
            kernel_new = kernel.resize((int(scale * w), int(scale * h)))
            kernel_new = np.asarray(kernel_new)
            kernels_out.append(kernel_new)
    return kernels_out

def buildKernels():
    json_path = './RedLights_annotation.json'
    data_path = '../data/RedLights2011_Medium'

    np.random.seed(1)
    file_names = sorted(os.listdir(data_path))
    file_names = [f for f in file_names if '.jpg' in f]
    np.random.shuffle(file_names)

    annotation = json.load(open(json_path,'r'))

    kernels = []

    for file_name in file_names:
        I = Image.open(os.path.join(data_path,file_name))
        I = np.asarray(I)

        boxes = annotation[file_name]

        for j in range(len(boxes)):
            tl_row, tl_col, br_row, br_col = boxes[j]
            K = I[tl_row: br_row, tl_col: br_col]
            kernels.append(K)

    kernels = kernels[:5]
    kernels = augmentedKernels(kernels)
    kernels = [k/np.linalg.norm(k) for k in kernels]

    return kernels

def convolution(arg):
    image = arg[0]
    kernel = arg[1]
    imageX = image.shape[0]
    imageY = image.shape[1]

    kernelX = kernel.shape[0]
    kernelY = kernel.shape[1]

    pad_width = ((int(kernelX/2), int(kernelX/2)), (int(kernelY/2), int(kernelY/2)), (0, 0))
    padded_image = np.pad(image, pad_width, mode='constant')

    outX = imageX
    outY = imageY
    out = np.zeros((outX, outY, 3))
    for i in range(outX):
        for j in range(outY):
            patch = padded_image[i:i+kernelX, j:j+kernelY]
            patch = patch/np.linalg.norm(patch)
            out[i, j] = (patch * kernel).sum()
    return out

def candidateRedLightHeatmap(image, kernels, threshold):
    imageX = image.shape[0]
    imageY = image.shape[1]

    pool = mp.Pool(mp.cpu_count())
    args = [(image, kernel) for kernel in kernels]
    convolved_list = pool.map(convolution, args)
    pool.close()
    pool.join()

    out = np.zeros((imageX, imageY, 3, len(kernels)))
    for i, kernel in enumerate(kernels):
        out[:,:,:,i] = convolved_list[i]

    single_channel = out.mean(axis=(2)) # average across RGB
    max_conv = single_channel.max(axis=(-1)) # max across kernels
    max_kernel_idxs = single_channel.argmax(axis=(-1))
    heatmap = max_conv > threshold

    return heatmap, max_kernel_idxs

def heatmap2boundingBoxes(heatmap, max_kernel_idxs, kernels):
    x_indices, y_indices = np.where(heatmap == 1.)
    idxs = list(zip(x_indices, y_indices))

    imageX = heatmap.shape[0]
    imageY = heatmap.shape[1]

    out = []
    for idx in idxs:
        x = idx[0]
        y = idx[1]
        kernel_idx = max_kernel_idxs[x, y]
        kernel = kernels[kernel_idx]
        kernelX = kernel.shape[0]
        kernelY = kernel.shape[1]
        tl_row = int(max(x - kernelX/2, 0))
        tl_col = int(max(y - kernelY/2, 0))
        br_row = int(min(x + kernelX/2, imageX))
        br_col = int(min(y + kernelY/2, imageY))
        out.append([tl_row, tl_col, br_row, br_col])

    return out

def detect_red_lights(image, kernels, threshold):
    heatmap, max_kernel_idxs = candidateRedLightHeatmap(image, kernels, threshold)
    bounding_boxes = heatmap2boundingBoxes(heatmap, max_kernel_idxs, kernels)
    return bounding_boxes

