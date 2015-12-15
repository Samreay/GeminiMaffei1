import os
import sys
import numpy as np
import matplotlib, matplotlib.pyplot as plt
import sextractor
from astropy.io import fits
from scipy.ndimage.filters import *
import copy
import scipy
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage.transform import resize
from time import time


debugFlag = True
debugPlot = True
sexPath = '/Users/shinton/Software/Ureka/bin/sex'
fitsPath = "../resources/clarisse+sky.fits"
cmap = "Blues_r"

def debug(msg):
    if debugFlag:
        print(msg)
        sys.stdout.flush()


def getFilters():
    dirName = "../resources/filters"
    filters = {}
    for f in os.listdir(dirName):
        if f.endswith(".conv"):
            filters[f[:-5]] = np.loadtxt(dirName + os.sep + f, skiprows=1)
    return filters
    
def getSextractor(**kwargs):
    # Create a SExtractor instance
    sex = sextractor.SExtractor()
    
    # Modify the SExtractor configuration
    sex.config['PIXEL_SCALE'] = 0.2
    sex.config['VERBOSE_TYPE'] = "QUIET"
    sex.config['MAG_ZEROPOINT'] = 25.0
    sex.config['BACK_SIZE'] = 8
    sex.config['BACK_TYPE'] = "AUTO"
    sex.config['BACKPHOTO_TYPE'] = "LOCAL"
    sex.config['BACK_FILTERSIZE'] = 3
    sex.config['SEEING_FWHM'] = 0.6
    sex.config['GAIN'] = 5.0
    sex.config['SATUR_LEVEL'] = 45000
    sex.config['MAG_GAMMA'] = 4.0
    sex.config['PHOT_APERTURES'] = [5, 8, 10]
    sex.config['CHECKIMAGE_TYPE'] = ["BACKGROUND"]#, "APERTURES"]
    sex.config['CHECKIMAGE_NAME'] = ["back.fits"]#, "apertures.fits"]
    sex.config['PARAMETERS_LIST'] = ['NUMBER', 'X_IMAGE', 'Y_IMAGE', 'MAG_APER(1)', 'MAG_AUTO', 'ELLIPTICITY', 'FWHM_IMAGE', 'MU_MAX', 'CLASS_STAR']
    for (key, value) in kwargs.iteritems():
        sex.config[key] = value
    return sex


def getInitialObjects():
    filters = getFilters()
    imageName = "../resources/clarisse+sky.fits"
    sex = getSextractor()
    sex.config['FILTER_MASK'] = filters['gauss_3.0_5x5']
    data = sex.run(imageName, path=sexPath)
    return data

def getDefaultImgComparison():
    fig = plt.figure(figsize=(14,10))
    ax0 = fig.add_subplot(1,2,1)
    ax1 = fig.add_subplot(1,2,2)
    ax0.set_title("Orginal data")
    ax1.set_title("Median subtracted")
    return fig,ax0,ax1

def addColourBars(inps):
    for handle, axis in inps:
        divider0 = make_axes_locatable(axis)
        cax0 = divider0.append_axes("right", size="4%", pad=0.05)
        cbar0 = plt.colorbar(handle, cax=cax0)
    
def getMaskedImage(imageStart):
    debug("Masking image")
    median = np.median(imageStart)
    badPixels = (imageStart < -1000) | (imageStart > 1e9)
    imageFixed = imageStart * (1 - badPixels) + median * badPixels
    
    if debugPlot:
        fig, ax0, ax1 = getDefaultImgComparison()
        ax0.set_title("Input image")
        ax1.set_title("Masked image")
        im0 = ax0.imshow(imageStart, vmin=1300, vmax=1400, cmap=cmap)
        im1 = ax1.imshow(imageFixed, vmin=1300, vmax=1400, cmap=cmap)
        addColourBars([(im0, ax0), (im1, ax1)])
        plt.show()
    debug("Image masked")
    return imageFixed
    
    
    
def broadClean(originalImage, redo=False, fast=False, s=100):
    outputName = "../resources/clarisseBroadClean.fits"
    analyse = True
    if not redo and os.path.isfile(outputName):
        try:
            f = fits.open(outputName)
            filterImage = f[0].data
            f.close()
            debug("Loaded existing file at %s" % outputName)
            analyse = False
        except Exception:
            debug("Previous file at %s not found" % outputName)
    if analyse:
        debug("Generating broad clean fits file")
        if fast:
            ratio = 0.5
            debug("Fast option selected. Reducing resolution by factor of %0.2f" % (1/ratio))
            s = np.ceil(s * ratio)
            image = resize(originalImage, np.floor(np.array(originalImage.shape) * ratio), order=1, preserve_range=True)
            debug("Image resized")
        else:
            image = originalImage
        y = np.arange(-s+1,s)[:,np.newaxis]
        x = np.arange(-s+1,s)[np.newaxis,:]
        r = np.sqrt(x*x + y*y)
        fil = ((r < s) & (r > np.floor(s * 0.7)))
        debug("Beginning convolution")
        t = time()
        filterImage = median_filter(image, footprint=fil)
        debug("Convolution completed in %0.2f seconds" % (time() - t))
        if fast:
            debug("Expanding image to original size")
            filterImage = resize(filterImage, originalImage.shape, order=1, preserve_range=True) #scipy.ndimage.interpolation.zoom
        
    if debugPlot:
        fig, ax0, ax1 = getDefaultImgComparison()
        ax0.set_title("Input image")
        ax1.set_title("Median filter image")
        im0 = ax0.imshow(originalImage, vmin=1310, vmax=1450, cmap=cmap)
        im1 = ax1.imshow(filterImage, vmin=1310, vmax=1450, cmap=cmap)
        addColourBars([(im0, ax0), (im1, ax1)])
        plt.show()
    if analyse:
        hdu = fits.PrimaryHDU(filterImage)
        if os.path.isfile(outputName):
            debug("Existing broad clean file found. Overwriting.")
        hdu.writeto(outputName, clobber=True)
        debug("Broad clean fits file generated and saved to %s" % outputName)
    return (outputName, filterImage)
    
    
def getSubtracted(originalImage, cleanImage):
    result = originalImage - cleanImage
    debug("Subtracting clean background")
    if debugPlot:
        fig, ax0, ax1 = getDefaultImgComparison()
        ax0.set_title("Input image")
        ax1.set_title("Background subtracted image")
        im0 = ax0.imshow(originalImage, vmin=1310, vmax=1450, cmap=cmap)
        im1 = ax1.imshow(result, vmin=0, vmax=100, cmap=cmap)
        addColourBars([(im0, ax0), (im1, ax1)])
        plt.show()
    debug("Background subtracted")
    return result
    
    
    
    
    
    
    
# Get the fits file
fitsFile = fits.open(fitsPath)
imageOriginal = fitsFile[0].data
fitsFile.close()

# Add mask to help out the median convolution
imageMasked = getMaskedImage(imageOriginal)
# Perform the median convolution to remove galaxy light pollution
(outputName, imageClean) = broadClean(imageMasked, redo=False, fast=True)
# Subtract out the light pollution
imageSubtracted = getSubtracted(originalImage, cleanImage)
# Find the correct sky flux
# TODO
# Add this back into the image, because some programs require it
# TODO













