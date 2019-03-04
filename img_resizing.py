import sys
import numba
from tqdm import trange
import numpy as np
from imageio import imread, imwrite
from scipy.ndimage.filters import convolve

def calc_energy(img):
    filter_du = np.array([
        [1.0, 2.0, 1.0],
        [0.0, 0.0, 0.0],
        [-1.0, -2.0, -1.0],
    ])
    # This converts it from a 2D filter to a 3D filter, replicating the same
    # filter for each channel: R, G, B
    filter_du = np.stack([filter_du] * 3, axis=2)

    filter_dv = np.array([
        [1.0, 0.0, -1.0],
        [2.0, 0.0, -2.0],
        [1.0, 0.0, -1.0],
    ])
    # This converts it from a 2D filter to a 3D filter, replicating the same
    # filter for each channel: R, G, B
    filter_dv = np.stack([filter_dv] * 3, axis=2)

    img = img.astype('float32')
    convolved = np.absolute(convolve(img, filter_du)) + np.absolute(convolve(img, filter_dv))

    # We sum the energies in the red, green, and blue channels
    energy_map = convolved.sum(axis=2)

    return energy_map

def crop_c(img, new_c):
    r, c, _ = img.shape
    for i in trange(c - new_c):        # use range if you don't want to use tqdm
        img = carve_column(img)
    return img

def crop_r(img, new_r):
    img = np.rot90(img, 1, (0, 1))
    img = crop_c(img, new_r)
    img = np.rot90(img, 3, (0, 1))
    return img

def crop_r_c(img,new_r,new_c):
    img = crop_r(img, new_r)
    img = crop_c(img, new_c)
    return img

@numba.jit
def carve_column(img):
    r, c, _ = img.shape

    M, backtrack = minimum_seam(img)
    mask = np.ones((r, c), dtype=np.bool)

    j = np.argmin(M[-1])
    for i in reversed(range(r)):
        mask[i, j] = False
        j = backtrack[i, j]

    mask = np.stack([mask] * 3, axis=2)
    img = img[mask].reshape((r, c - 1, 3))
    return img

@numba.jit
def minimum_seam(img):
    r, c, _ = img.shape
    energy_map = calc_energy(img)

    M = energy_map.copy()
    backtrack = np.zeros_like(M, dtype=np.int)

    for i in range(1, r):
        for j in range(0, c):
            # Handle the left edge of the image, to ensure we don't index a -1
            if j == 0:
                idx = np.argmin(M[i-1, j:j + 2])
                backtrack[i, j] = idx + j
                min_energy = M[i-1, idx + j]
            else:
                idx = np.argmin(M[i - 1, j - 1:j + 2])
                backtrack[i, j] = idx + j - 1
                min_energy = M[i - 1, idx + j - 1]

            M[i, j] += min_energy

    return M, backtrack

def main():
    in_filename = input("Enter the input file name(Ex:- 123.jpg) : ")
    img = imread(in_filename)
    old_r,old_c,_=img.shape
    print("Original Dimension of input file : ",old_r,old_c,_)
    new_r,new_c=input("Enter Value New Dimension(Ex:- New_Row_Value New_Column_Value) : ").split()
    new_r=int(new_r)
    new_c=int(new_c)
    if new_r>=old_r and new_c>=old_c:
        print("Enter the value less than original dimension")
        sys.exit(1)
    out_filename = input("Enter the output file name(Ex:- 456.jpg) : ")


    if old_r!=new_r and old_c!=new_c:
        out=crop_r_c(img,new_r,new_c)
    elif old_r!=new_r:
        out = crop_r(img, new_r)
    elif old_c!=new_c:
        out = crop_c(img, new_c)
    
    imwrite(out_filename, out)

if __name__ == '__main__':
    main()
