import glob
from skimage import io

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import os
def fourier(img):

    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    out = 20*np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

    # inverse_shift = np.fft.fftshift(dft_shift)
    # inverse_dft = cv2.dft(inverse_shift, flags=cv2.DFT_INVERSE)
    # out2 = cv2.magnitude(inverse_dft[:, :, 0], inverse_dft[:, :, 1])

    return out
def npy2fimages(input_dir, output_dir):
    """Resize the images in 'input_dir' and save into 'output_dir'."""
    for idir in os.scandir(input_dir):
        # print(output_dir + '/' + idir.name)
        f = glob.glob(idir.path)[0]
        img = io.imread(f, as_gray=True)
        img = fourier(img)
        y,x = img.shape
        startx = x // 2 - (224 // 2)
        starty = y // 2 - (224 // 2)
        crop_img = img[starty:starty+224,startx:startx+224]
        name = idir.name

        cv2.imwrite(os.path.join(output_dir + '/', name + '.jpg'), crop_img)

        # if (iimage + 1) % 1000 == 0:
        #     print("[{}/{}] fourier transformed images and saved into '{}'."
        #           .format(iimage + 1, n_images, output_dir + '/' + idir.name))
# define a main function
def main():
    input_dir = 'D:/data/iuct/images/images_normalized'
    output_dir = 'D:/data/iuct/images/fourier'
    npy2fimages(input_dir, output_dir)


if __name__ == '__main__':

    main()
    # file_path = 'D:/data/brain/preprocessed_image_data/preprocessed_data/fourier_cut_cropped_img/cropped_img'
    # file_names=os.listdir(file_path)
    #
    #
    # for name in file_names:
    #     src = os.path.join(file_path, name)
    #     nname = name.split('(')[0]
    #     dst = nname + '.jpg'
    #     dst = os.path.join(file_path, dst)
    #     os.rename(src, dst)

