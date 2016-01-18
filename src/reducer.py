import os
import sys
import tempfile
import shutil
import errno

import numpy as np
import matplotlib, matplotlib.pyplot as plt
import scipy

from time import time
from numpy.lib.recfunctions import append_fields
from scipy.interpolate import interp1d
from scipy.ndimage.filters import *
from scipy.ndimage.filters import *
from skimage.transform import resize
from astropy.io import fits
from mpl_toolkits.axes_grid1 import make_axes_locatable

import sextractor


urekaPath = "/Users/shinton/Software/Ureka"
wcsToolsPath = "/Users/shinton/Software/wcstools/bin"
baolabScript = '/Users/shinton/Software/baolab-0.94.1e/als2bl.cl'
sexPath = '/Users/shinton/Software/Ureka/bin/sex'
scampPath = '/usr/local/bin/scamp'
missfitsPath = '/usr/local/bin/missfits' 

class Reducer(object):
    def __init__(self, tempParentDir=None, debug=True, debugPlot=False, tempSubDir="", redo=False):
        self.tempParentDir = tempParentDir
        self.tempSubDir = tempSubDir
        self.debug = debug
        self.debugPlot = debugPlot
        self.tempDir = None
        self.outDir = "../out" + os.sep + tempSubDir
        self.cmap = "Blues_r"
        self.redo = redo
        self.name = None
        self.tempDirSetup = False
        
    def _debug(self, message):
        if self.debug:
            print(message)
            sys.stdout.flush()
        
    def _cleanTempDir(self):
        try:
            shutil.rmtree(self.tempDir)
            pass
        except OSError as exc:
            if exc.errno != errno.ENOENT:
                raise
        
    def _setupTempDir(self):
        if self.tempParentDir is None:
            self.tempParentDir = tempfile.mkdtemp()
        self.tempDir = self.tempParentDir + os.sep + self.tempSubDir
        self._cleanTempDir()
        os.mkdir(self.tempDir)
        if not os.path.exists(self.outDir):
            os.mkdir(self.outDir)
        self.tempDirSetup = True
        self._debug("Generating temp directory at %s for classifier" % self.tempDir)

    def getCatalog(self, fitsPath):   
        self.name = os.path.splitext(os.path.basename(fitsPath))[0]
        catalogOutFile = self.outDir + os.sep + self.name + "Cat.npy"
        if not self.redo and os.path.exists(catalogOutFile):
            self._debug("Loading existing catalog")
            self.catalog = np.load(catalogOutFile)
            return self.catalog
              
        if not self.tempDirSetup: 
            self._setupTempDir()
            
        self._debug("Copying fits file to temp directory")
   

        path = self.tempDir + os.sep + self.name
        shutil.copy(fitsPath, path)
        imageOriginal = fits.getdata(path)
        mask, imageMasked = self.getMaskedImage(imageOriginal)
        imageSubtracted = self.getBackground(imageMasked)
        catalog, sex = self._getCatalogs(fitsPath, imageSubtracted)
        catalogTrimmed = self.trimCatalog(catalog, imageOriginal, mask, sex)
        catalogFinal = self.normaliseRadial(catalogTrimmed, sex)
        np.save(catalogOutFile, catalogFinal)
        self.catalog = catalogFinal
        return catalogFinal

    def normaliseRadial(self, catalog, sex):
        
        magArray = []
        self._debug("Constructing magnitude arrays")
        for entry in catalog:
            mags = [entry['MAG_APER(%d)'%(i+1)] for i in range(len(sex.config['PHOT_APERTURES']))]
            magArray.append(mags)
        magArray = np.array(magArray)
        magArrayNormalised = magArray - np.max(magArray, axis=1)[np.newaxis].T
        
        mins = np.min(magArrayNormalised, axis=1)
        magArrayNormalised = magArrayNormalised / mins[np.newaxis].T
        radii = sex.config['PHOT_APERTURES']
        
        result = []
        t = 1/np.sqrt(2)
        for row in magArrayNormalised:
            pixels = interp1d(row, radii, kind="linear")(t)
            result.append(np.round(pixels,3))
        result = np.array(result)
        
        self._debug("Removing 'MAG_APER' from catalog, replacing with 'FALLOUT'")
        names = [n for n in catalog.dtype.names if "MAG_APER" not in n]
        catalogFinal = catalog[names].copy()
        catalogFinal = append_fields(catalogFinal, 'FALLOFF', result, usemask=False)    
        
        
        '''if debugPlot:
            debug("Creating radial plot")    
            fig = plt.figure(figsize=(15,5))
            ax0 = fig.add_subplot(1,3,1)
            ax1 = fig.add_subplot(1,3,2)
            ax2 = fig.add_subplot(1,3,3)
            for i,row in enumerate(magArrayNormalised):
                if maskExtended[i]:
                    alpha = 1
                    c = 'r'
                else:
                    if np.random.random() > 0.9:
                        alpha = 0.05
                        c = 'b'
                    else:
                        continue
                ax0.plot(radii, row, color=c, alpha=alpha)
                
            d = np.diff(radii)
            bins = np.concatenate(([radii[0] - d[0]], radii[1:] + 0.5*d))
            h1, ed1 = np.histogram(result[maskExtended], bins=bins, density=False)
            x1 = 0.5*(ed1[1:] + ed1[:-1])
    
            h2, ed2 = np.histogram(result[~maskExtended], bins=bins, density=False)
            x2 = 0.5*(ed2[1:] + ed2[:-1])
            width= 0.7 * diff(bins)
            ax1.bar(x1, 1.0*h1/h1.sum(), width=width, label="Extended", color='r', alpha=0.5)
            ax1.bar(x2, 1.0*h2/h2.sum(), width=width, label="Point", color='b', alpha=0.5)
            ax2.plot(x1, h1, label="Extended", color='r')
            ax2.plot(x2, h2, label="Point", color='b')
            ax2.set_yscale('log')
            ax1.set_xlabel("Pix at 0.707 normalised magnitude (Prob)")
            ax2.set_xlabel("Pix at 0.707 normalised magnitude (Number)")
            ax0.set_xlabel("Pix")
            ax0.set_ylabel("Normalised magnitude falloff")
            ax1.set_xlim(2, 8)
            ax1.legend()
            ax2.legend()
            plt.show()'''
    
        return catalogFinal
        
    def _getCatalogs(self, fitsPath, image, pixelThreshold=2):
        self._debug("Getting objects using sextractor")
        tempFits = os.path.abspath(self.tempDir + os.sep + "temp_%s"%self.name)
        
        fitsFile = fits.open(fitsPath)
        fitsFile[0].data = image
        fitsFile.writeto(tempFits)
        fitsFile.close()
        
        filters = self.getFilters()
        sex = self.getSextractor()
        sex.config['FILTER_MASK'] = filters['gauss_4.0_7x7']
        self._debug("Running sextractor using gaussian filter")
        data1 = sex.run(tempFits, path=sexPath, cwd=self.tempDir)
        sex.config['FILTER_MASK'] = filters['mexhat_2.5_7x7']
        self._debug("Running sextractor using mexhat filter")
        data2 = sex.run(tempFits, path=sexPath, cwd=self.tempDir)
        
        cat1 = data1['catalog']
        cat2 = data2['catalog']
        
        self._debug("Merging catalogs")
        mask = []
        for entry in cat2:
            x = entry['X_IMAGE']
            y = entry['Y_IMAGE']
            xdiff = cat1['X_IMAGE'] - x
            xdiff *= xdiff
            ydiff = cat1['Y_IMAGE'] - y
            ydiff *= ydiff
            rdiff = xdiff + ydiff
            rmin= rdiff.min()
            mask.append(rmin > pixelThreshold)
        mask = np.array(mask)
        catTotal = np.concatenate((cat1, cat2[mask]))
        self._debug("Catalogs merged. Found %d objects." % catTotal.size)
    
        return catTotal, sex
        
    def trimCatalog(self, catalog, imageOriginal, maskBad, sex, border=60, expand=3, magLimit=30):
        self._debug("Trimming catalog")
        maskBad = maskBad * 1
        maskBad[:,0:border] = 1
        maskBad[:, -border:] = 1
        maskBad[0:border, :] = 1
        maskBad[-border:,:] = 1
        expandConvolve = np.ones((5,5))
        
        ratio = 0.25
        maskShrunk = resize(maskBad, np.floor(np.array(maskBad.shape) * ratio), order=1, preserve_range=True)
        
        self._debug("Expanding bad pixel mask")
        for i in range(expand):
            maskShrunk = scipy.signal.convolve2d(maskShrunk, expandConvolve, mode="same")
            
        maskBad = resize(maskShrunk, maskBad.shape, order=1, preserve_range=True)       
        maskGood = (maskBad == 0)
        
        catalogTrimmed = catalog[(maskGood[catalog['Y_IMAGE'].astype(int) - 1,catalog['X_IMAGE'].astype(int) - 1])]
        
        self._debug("Removing sources that are too faint")
        for i in range(len(sex.config['PHOT_APERTURES'])):
            catalogTrimmed = catalogTrimmed[catalogTrimmed['MAG_APER(%d)'%(i+1)] < magLimit]
        if self.debugPlot:
            fig, ax0, ax1 = self.getDefaultImgComparison()
            ax0.set_title("Input image and catalog")
            ax1.set_title("Trimmed")
            im0 = ax0.imshow(imageOriginal, vmin=1310, vmax=1450, cmap=self.cmap, origin="lower")
            ax0.scatter(catalog['X_IMAGE'], catalog['Y_IMAGE'], color='r', s=1)
            im1 = ax1.imshow(imageOriginal, vmin=1310, vmax=1450, cmap=self.cmap, origin="lower")
            ax1.scatter(catalogTrimmed['X_IMAGE'], catalogTrimmed['Y_IMAGE'], color='r', s=1)
            self.addColourBars([(im0, ax0), (im1, ax1)])
            plt.show()
        self._debug("Trimming went from %d objects to %d objects" % (catalog.size, catalogTrimmed.size))
        return catalogTrimmed
        
    def getFilters(self):
        dirName = "../resources/filters"
        filters = {}
        for f in os.listdir(dirName):
            if f.endswith(".conv"):
                filters[f[:-5]] = np.loadtxt(dirName + os.sep + f, skiprows=1)
        return filters
        
    def getDefaultSextractor(self):
        # Create a SExtractor instance
        sex = sextractor.SExtractor()
                
        
        # Modify the SExtractor configuration
        sex.config['PIXEL_SCALE'] = 0.2
        sex.config['VERBOSE_TYPE'] = "QUIET"
        sex.config['MAG_ZEROPOINT'] = 25.0
        sex.config['BACK_SIZE'] = 256
        sex.config['BACK_TYPE'] = "MANUAL"
        #sex.config['BACK_VALUE'] = 0
        sex.config['BACKPHOTO_TYPE'] = "LOCAL"
        sex.config['BACK_FILTERSIZE'] = 3
        sex.config['DETECT_THRESH'] = 2
        sex.config['ANALYSIS_THRESH'] = 2
        sex.config['SEEING_FWHM'] = 0.6
        sex.config['GAIN'] = 5.0
        sex.config['SATUR_LEVEL'] = 45000
        sex.config['MAG_GAMMA'] = 4.0
        sex.config['MEMORY_OBJSTACK'] = 30000
        sex.config['MEMORY_PIXSTACK'] = 6000000
        sex.config['MEMORY_BUFSIZE'] = 16384
        sex.config['DEBLEND_MINCONT'] = 0.001
        
        if self.fitsPath is not None:
            potSeeing = os.path.dirname(self.fitsPath) + os.sep + "seeing_fwhm"
            if os.path.exists(potSeeing):
                res = np.genfromtxt(potSeeing, dtype=None)
                for f,s in res:
                    if f == os.path.basename(self.fitsPath):
                        sex.config['SEEING_FWHM'] = sex.config['PIXEL_SCALE'] * s
                        self._debug("seeing fwhm updated to be %0.2f" % sex.config['SEEING_FWHM'])
        return sex
    
    def getScampSextractor(self):
        sex = self.getDefaultSextractor()
        sex.config['CATALOG_TYPE'] = 'FITS_LDAC'
        sex.config['PARAMETERS_LIST'] = ['NUMBER','X_IMAGE','Y_IMAGE','XWIN_IMAGE','YWIN_IMAGE','ELLIPTICITY','ERRAWIN_IMAGE','ERRBWIN_IMAGE','ERRTHETAWIN_IMAGE','FLUX_AUTO','FLUXERR_AUTO','FLAGS','FLAGS_WEIGHT','FLUX_RADIUS']
        filters = self.getFilters()
        sex.config['FILTER_MASK'] = filters['gauss_3.0_5x5']
    
        return sex
    
    def getSextractor(self, **kwargs):
        sex = self.getDefaultSextractor()
        sex.config['PHOT_APERTURES'] = [1,2,2.5,3,3.25,3.5,3.75,4,4.25,4.5,5,5.5,6,6.5,7,8,12,14,18,20,25]
        sex.config['CHECKIMAGE_TYPE'] = ["BACKGROUND", "APERTURES"]
        sex.config['CHECKIMAGE_NAME'] = ["back.fits", "apertures.fits"]
        sex.config['PARAMETERS_LIST'] = ['NUMBER', 'X_IMAGE', 'Y_IMAGE'] + ['MAG_APER(%d)'%(i+1) for i in range(len(sex.config['PHOT_APERTURES']))] + ['FLUX_MAX', 'FLUX_AUTO', 'MAG_AUTO', 'ELLIPTICITY', 'FWHM_IMAGE', 'MU_MAX', 'CLASS_STAR']
        for (key, value) in kwargs.iteritems():
            sex.config[key] = value
        return sex    
        
        
        
    def getMaskedImage(self, imageStart):
        self._debug("Masking image")
        median = np.median(imageStart)
        badPixels = (imageStart < -1000) | (imageStart > 1e6)
        imageMasked = imageStart * (1 - badPixels) + median * badPixels
        
        if self.debugPlot:
            fig, ax0, ax1 = self.getDefaultImgComparison()
            ax0.set_title("Input image")
            ax1.set_title("Masked image")
            im0 = ax0.imshow(imageStart, vmin=1300, vmax=1400, cmap=self.cmap, origin="lower")
            im1 = ax1.imshow(imageMasked, vmin=1300, vmax=1400, cmap=self.cmap, origin="lower")
            self.addColourBars([(im0, ax0), (im1, ax1)])
            plt.show()
        self._debug("Image masked")
        return badPixels, imageMasked

    def getDefaultImgComparison(self):
        fig = plt.figure(figsize=(14,10))
        ax0 = fig.add_subplot(1,2,1)
        ax1 = fig.add_subplot(1,2,2)
        return fig,ax0,ax1

    def addColourBars(self, inps):
        for handle, axis in inps:
            divider0 = make_axes_locatable(axis)
            cax0 = divider0.append_axes("right", size="4%", pad=0.05)
            plt.colorbar(handle, cax=cax0)

    def getBackground(self, originalImage, size=150):
        outputName = self.outDir + os.sep + self.name + "Background.fits"
        if not self.redo and os.path.exists(outputName):
            self._debug("Loading existing background image from %s" % outputName)
            imageBackground = fits.getdata(outputName)
            return originalImage - imageBackground
            
        ratio = 0.25
        self._debug("Redoing background. Resizing image by factor of %0.2f" % ratio)
        s = np.ceil(size * ratio)
        image = resize(originalImage, np.floor(np.array(originalImage.shape) * ratio), order=1, preserve_range=True)
        self._debug("Image resized") 
        
        y = np.arange(-s+1,s)[:,np.newaxis]
        x = np.arange(-s+1,s)[np.newaxis,:]
        r = np.sqrt(x*x + y*y)
        fil = ((r < s) & (r > np.floor(s * 0.7)))
        
        self._debug("Beginning convolution")
        t = time()
        filterImage = median_filter(image, footprint=fil)
        self._debug("Convolution completed in %0.2f seconds" % (time() - t))
        
        self._debug("Expanding image to original size")
        filterImage = resize(filterImage, originalImage.shape, order=1, preserve_range=True)
        
        if self.debugPlot:
            fig, ax0, ax1 = self.getDefaultImgComparison()
            ax0.set_title("Input image")
            ax1.set_title("Median filter image")
            im0 = ax0.imshow(originalImage, vmin=1310, vmax=1450, cmap=self.cmap, origin="lower")
            im1 = ax1.imshow(filterImage, vmin=1310, vmax=1450, cmap=self.cmap, origin="lower")
            self.addColourBars([(im0, ax0), (im1, ax1)])
            plt.show()
            
        hdulist = fits.PrimaryHDU(filterImage)
        hdulist.writeto(outputName, clobber=True)
        self._debug("Background fits file generated and saved to %s" % outputName)
        return originalImage - filterImage
        