import os
import sys
import numpy as np
import matplotlib, matplotlib.pyplot as plt
import sextractor
from astropy.io import fits
from scipy.ndimage.filters import *
import copy
import scipy

debugFlag = True
sexPath = '/Users/shinton/Software/Ureka/bin/sex'
fitsPath = "../resources/clarisse+sky.fits"

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


def broadClean(filename, redo=False, fast=False):
    outputName = "../resources/clarisseBroadClean.fits"
    if not redo and os.path.isfile(outputName):
        try:
            f = fits.open(outputName)
            debug("Loaded existing file at %s" % outputName)
            return (outputName, f[0].data)
        except Exception:
            debug("Previous file at %s not found" % outputName)
    debug("Generating broad clean fits file")
    fitsFile = fits.open(filename)
    image = fitsFile[0].data
    originalSize = image.size
    s = 100
    
    if fast:
        ratio = 0.25
        debug("Fast option selected. Reducing resolution by factor of %0.2f" % (1/ratio))
        s = np.ceil(s * ratio)
        image = scipy.misc.imresize(image, ratio, interp="nearest")
        
    y = np.arange(-s+1,s)[:,np.newaxis]
    x = np.arange(-s+1,s)[np.newaxis,:]
    r = np.sqrt(x*x + y*y)
    fil = ((r < s) & (r > np.floor(s * 0.7)))
    
    filterImage = median_filter(image, footprint=fil)
    
    if fast:
        debug("Expanding image to original size")
        filterImage = scipy.misc.imresize(filterImage, originalSize)
    
    fig = plt.figure(figsize=(14,10))
    ax0 = fig.add_subplot(1,2,1)
    ax1 = fig.add_subplot(1,2,2)
    ax0.set_title("Orginal data")
    ax1.set_title("Median subtracted")
    
    ax0.imshow(image, vmin=1310, vmax=1450, cmap="Blues_r")
    ax1.imshow(filterImage, vmin=1310, vmax=1450, cmap="Blues_r")
    
    fitsFile.data = filterImage
    if os.path.isfile(outputName):
        debug("Existing broad clean file found. Overwriting.")
    fitsFile.writeto(outputName, clobber=True)
    fitsFile.close()
    debug("Broad clean fits file generated and saved to %s" % outputName)
    return (outputName, filterImage)
    
(outputName, cleanImage) = broadClean(fitsPath, redo=True, fast=True)












