import numpy as np
from PIL import Image
import pydicom
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
        if not idir.is_dir():
            continue
        if not os.path.exists(output_dir + '/' + idir.name):
            os.makedirs(output_dir + '/' + idir.name)
        npy_images = os.listdir(idir.path)
        n_images = len(npy_images)
        for iimage, npy_image in enumerate(npy_images):
            try:
               with open(os.path.join(idir.path, npy_image), 'r+b') as f:
                    #with np.load(f.name) as img:
                        img=np.load(f.name)
                        img = Image.fromarray(img)
                        img = fourier(img)
                        # y,x = img.shape
                        # startx = x // 2 - (16 // 2)
                        # starty = y // 2 - (16 // 2)
                        # crop_img = img[starty:starty+16,startx:startx+16]
                        name = npy_image.rstrip('.npy')

                        cv2.imwrite(os.path.join(output_dir + '/' + idir.name, name+'.jpg'), img)
                        #img.save(os.path.join(output_dir + '/' + idir.name, name+'.jpg'), img.format)
            except(IOError, SyntaxError) as e:
                pass
            if (iimage + 1) % 1000 == 0:
                print("[{}/{}] fourier transformed images and saved into '{}'."
                      .format(iimage + 1, n_images, output_dir + '/' + idir.name))
# define a main function
def main():
    input_dir = 'D:/data/brain/preprocessed_data_v2/cropped_img'
    output_dir = 'D:/data/brain/preprocessed_data_v2/fourier_img'
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

