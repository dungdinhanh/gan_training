import os
import csv
import numpy as np
from glob import glob
from scipy.misc import imread, imsave, imresize
from tqdm import tqdm


MNIST = "large_files/train.csv"


def get_mnist():
    current_folder = os.path.dirname(os.path.abspath(__file__))
    wd = os.path.abspath(os.path.join(current_folder, os.pardir))
    data_path = os.path.join(wd, MNIST)
    csv_file = open(data_path, 'r')
    csv_pointer = csv.reader(csv_file)
    list_mnist = []
    for row in csv_pointer:
        list_mnist.append(np.asarray(row))
    list_mnist = np.asarray(list_mnist[1:], dtype=int)
    Y = list_mnist[:,0]
    X = list_mnist[:, 1:]
    return X/255.0, Y


def get_celeb(limit=None):
    filenames = glob("../large_files/img_align_celeba/*.jgp")
    N = len(filenames)
    print("Found %d files!"%N)

    os.mkdir("../large_files/img_align_celba-cropped")
    print("Cropping images, please wait...")

    for i in range(N):
        crop_and_resave(filenames[i], '../large_files/img_align_celeba-cropped')
        if i % 1000 == 0:
            print("%d/%d"%(i, N))

    # make sure to return the cropped version
    filenames = glob("../large_files/img_align_celeba-cropped/*.jpg")
    return filenames


def crop_and_resave(inputfile, outputdir):
    im = imread(inputfile)
    height, width, color = im.shape
    edge_h = int( round( (height - 108)/ 2.0))
    edge_w = int( round( (width - 108)/ 2.0))

    cropped = im[edge_h: (edge_h + 108), edge_w: (edge_w + 108)]
    small = imresize(cropped, (64, 64))
    filename = inputfile.split('/')[-1]
    imsave("%s/%s"%(outputdir, filename), small)
    pass


def scale_image(im):
    return (im / 255.0)*2 - 1


def files2image(filenames):
    return [scale_image(imread(fn)) for fn in filenames]


def save_response_content(r, dest):
    total_iters = 1409659
    print("Note: units are in KB, e.g. KKB = MB")
    # because we are reading 1024 bytes at a time, hence
    # 1KB == 1 "unit" for tqdm
    with open(dest, "wb") as f:
        for chunk  in tqdm(
            r.iter_content(1024),
            total = total_iters,
            unit= 'KB',
            unit_scale=True):
            if chunk:
                f.write(chunk)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None









