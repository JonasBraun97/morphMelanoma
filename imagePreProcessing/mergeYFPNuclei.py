import numpy as np
import argparse
import cv2
import skimage.io
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
import os
from PIL import Image
from skimage import img_as_ubyte, exposure

os.chdir('/Users/jones/Library/CloudStorage/OneDrive-NorthwesternUniversity/ownProject/data/rawData/clampFISCHIan/Subregion_45_r5_c5')
image_output = 'merged_DAPI_YFP_test.tif'
dapi = skimage.io.imread('R4_DAPI.tif')
yfp = skimage.io.imread('R4_YFP.tif')

DAPI_8bit_NormalizeScikit = img_as_ubyte(exposure.rescale_intensity(dapi))
YFP_8bit_NormalizeScikit = img_as_ubyte(exposure.rescale_intensity(yfp))

zeros = np.zeros(dapi.shape[:2], dtype="uint8")
merged8_NormalizeScikit = cv2.merge([zeros, YFP_8bit_NormalizeScikit, DAPI_8bit_NormalizeScikit])

im = Image.fromarray(merged8_NormalizeScikit)
im.save('YFP_DAPI_rgb8bitScikit.tif')
