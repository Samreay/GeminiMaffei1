import os
import sys
import tempfile
import shutil
import errno
import re
import subprocess

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

    def getCatalog(self, fitsPath, ishape=False, excluded=None):
        self.name = os.path.splitext(os.path.basename(fitsPath))[0]
        catalogOutFile = self.outDir + os.sep + self.name + ("_ishape_" if ishape else "") + "Cat.npy"
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
        if excluded is not None:
            self._debug("Removing %d excluded sources. Had %d sources."%(excluded.shape[0],catalog.shape[0]))
            catalog = catalog[~self._getCandidateMask(catalog, excluded[['X_IMAGE','Y_IMAGE']].view(np.float64).reshape(excluded.shape + (-1,)))]
            self._debug("Went to %d sources"%catalog.shape[0])
        catalogTrimmed = self.trimCatalog(catalog, imageOriginal, mask, sex)
        catalogFinal = self.normaliseRadial(catalogTrimmed, sex)
        self.catalog = catalogFinal
        
        if ishape:
            #Get ishape data now
            ishapeRes = self.runIshape(fitsPath, catalogFinal, np.ones(catalogFinal.shape) == 1)            
            catalogFinal = self.addIshapeDetails(catalogFinal, ishapeRes)
            
        np.save(catalogOutFile, catalogFinal)
        return catalogFinal
        
    def _getCandidateMask(self, catalog, coord, pixelThreshold=4):
        self._debug("Loading in extendeds")
        mask = []
        for entry in catalog:
            dists = np.sqrt((entry['X_IMAGE'] - coord[:,0])**2 + (entry['Y_IMAGE'] - coord[:,1])**2 )
            mask.append(dists.min() < pixelThreshold)
        mask = np.array(mask)
        return mask
    def addIshapeDetails(self, catalog, parsed):        
        self._debug("Adding in ishape results")
        kingFwhm = []
        chi2Kings = []
        chi2Deltas = []
        divs = []
        subs = []
        newMask = []
        for i,row in enumerate(catalog):
            p = self.getMatch(row, parsed)
            if p is not None:
                chi2King = p['KING30'][3]
                chi2Delta = p['KING30'][4]
                fwhm = p['KING30'][0]
                div = chi2Delta/chi2King
                sub = chi2Delta - chi2King
                if np.isfinite(chi2King) and np.isfinite(chi2Delta) and np.isfinite(fwhm) and np.isfinite(div) and np.isfinite(sub):    
                    newMask.append(True)
                    divs.append(np.round(div, decimals=6))
                    subs.append(np.round(sub, decimals=6))
                    chi2Kings.append(np.round(chi2King, decimals=6))
                    chi2Deltas.append(np.round(chi2Delta, decimals=6))
                    kingFwhm.append(fwhm)
                else:
                    newMask.append(False)
            else:
                newMask.append(False)
        newMask = np.array(newMask)
        catalogNew = catalog[newMask]
        catalogNew = append_fields(catalogNew, 'Chi2DeltaKingDiv', divs, usemask=False)    
        catalogNew = append_fields(catalogNew, 'Chi2DeltaKingSub', subs, usemask=False)    
        catalogNew = append_fields(catalogNew, 'Chi2King', chi2Kings, usemask=False)    
        catalogNew = append_fields(catalogNew, 'Chi2Delta', chi2Deltas, usemask=False)    
        catalogNew = append_fields(catalogNew, 'KingFWHM', kingFwhm, usemask=False)
        
        self._debug("\tNew catalog has %d gcs, compared to %d previously" % (newMask.sum(), catalog.shape[0]))
        return catalogNew

    def normaliseRadial(self, catalog, sex):
        
        magArray = []
        self._debug("Constructing magnitude arrays")
        for entry in catalog:
            mags = [entry['MAG_APER(%d)'%(i+1)] for i in range(len(sex.config['PHOT_APERTURES']))]
            magArray.append(mags)
        magArray = np.array(magArray)
        ci = magArray[:,0] - magArray[:,2]
        ci2 = magArray[:,1] - magArray[:,4]
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
        #catalogFinal = catalog.copy()
        catalogFinal = append_fields(catalogFinal, 'FALLOFF', result, usemask=False)    
        catalogFinal = append_fields(catalogFinal, 'CI', np.round(ci, decimals=6), usemask=False)    
        catalogFinal = append_fields(catalogFinal, 'CI2', np.round(ci2, decimals=6), usemask=False)    
        
        
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
        
    def _getCatalogs(self, fitsPath, image, pixelThreshold=2, aperture=False):
        self._debug("Getting objects using sextractor")
        tempFits = os.path.abspath(self.tempDir + os.sep + "temp_%s"%self.name)
        
        fitsFile = fits.open(fitsPath)
        if image is not None:
            fitsFile[0].data = image
        fitsFile.writeto(tempFits, clobber=True)
        fitsFile.close()
        
        filters = self.getFilters()
        if aperture:
            sex = self.getApertureSextractor()
            sex.config['FILTER_MASK'] = filters['gauss_3.0_7x7']
        else:
            sex = self.getSextractor()
            sex.config['FILTER_MASK'] = filters['gauss_3.0_7x7']
        self._debug("Running sextractor using gaussian filter")
        data1 = sex.run(tempFits, path=sexPath, cwd=self.tempDir)
        print(data1['catalog'].shape)
        sex.config['FILTER_MASK'] = filters['mexhat_2.5_7x7']
        self._debug("Running sextractor using mexhat filter")
        data2 = sex.run(tempFits, path=sexPath, cwd=self.tempDir)
        
        cat1 = data1['catalog']
        cat2 = data2['catalog']


        print(cat2.shape)
        self._debug("Merging catalogs")
        mask = []
        print(cat1.size)
        
        if cat1.size <= 1:
            catTotal = cat2
        else:
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
            im0 = ax0.imshow(imageOriginal, vmin=np.median(imageOriginal)-100, vmax=np.median(imageOriginal)+500, cmap=self.cmap, origin="lower")
            ax0.scatter(catalog['X_IMAGE'], catalog['Y_IMAGE'], color='r', s=1)
            im1 = ax1.imshow(imageOriginal, vmin=np.median(imageOriginal)-100, vmax=np.median(imageOriginal)+500, cmap=self.cmap, origin="lower")
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
        sex.config['MEMORY_PIXSTACK'] = 9000000
        sex.config['MEMORY_BUFSIZE'] = 16384
        sex.config['DEBLEND_MINCONT'] = 0.001
        
        try:
            if self.fitsPath is not None:
                potSeeing = os.path.dirname(self.fitsPath) + os.sep + "seeing_fwhm"
                if os.path.exists(potSeeing):
                    res = np.genfromtxt(potSeeing, dtype=None)
                    for f,s in res:
                        if f == os.path.basename(self.fitsPath):
                            sex.config['SEEING_FWHM'] = sex.config['PIXEL_SCALE'] * s
                            self._debug("seeing fwhm updated to be %0.2f" % sex.config['SEEING_FWHM'])
        except Exception:
            sex.config['SEEING_FWHM'] = sex.config['PIXEL_SCALE'] * 3.2
        return sex
    
    def getScampSextractor(self):
        sex = self.getDefaultSextractor()
        sex.config['CATALOG_TYPE'] = 'FITS_LDAC'
        sex.config['PARAMETERS_LIST'] = ['NUMBER','X_IMAGE','Y_IMAGE','XWIN_IMAGE','YWIN_IMAGE','ELLIPTICITY','ERRAWIN_IMAGE','ERRBWIN_IMAGE','ERRTHETAWIN_IMAGE','FLUX_AUTO','FLUXERR_AUTO','FLAGS','FLAGS_WEIGHT','FLUX_RADIUS']
        filters = self.getFilters()
        sex.config['FILTER_MASK'] = filters['gauss_3.0_5x5']
    
        return sex
    
    def getApertureSextractor(self, **kwargs):
        sex = self.getDefaultSextractor()
        sex.config['PHOT_APERTURES'] = [1]
        sex.config['PARAMETERS_LIST'] = ['NUMBER','X_IMAGE','Y_IMAGE', 'ELLIPTICITY','MAG_BEST','MAGERR_BEST']
        sex.config['CHECKIMAGE_TYPE'] = ["NONE"]
        sex.config['DETECT_THRESH'] = 2
        sex.config['ANALYSIS_THRESH'] = 2
        sex.config['CHECKIMAGE_NAME'] = []
        for (key, value) in kwargs.iteritems():
            sex.config[key] = value
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
            im0 = ax0.imshow(imageStart, vmin=np.median(imageStart)-100, vmax=np.median(imageStart)+500, cmap=self.cmap, origin="lower")
            im1 = ax1.imshow(imageMasked, vmin=np.median(imageMasked)-100, vmax=np.median(imageStart)+500, cmap=self.cmap, origin="lower")
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
            im0 = ax0.imshow(originalImage, vmin=np.median(originalImage)-100, vmax=np.median(originalImage)+500, cmap=self.cmap, origin="lower")
            im1 = ax1.imshow(filterImage, vmin=np.median(filterImage)-100, vmax=np.median(filterImage)+500, cmap=self.cmap, origin="lower")
            self.addColourBars([(im0, ax0), (im1, ax1)])
            plt.show()
            
        hdulist = fits.PrimaryHDU(filterImage.astype(np.float32))
        hdulist.writeto(outputName, clobber=True)
        self._debug("Background fits file generated and saved to %s" % outputName)
        return originalImage - filterImage
        
    def getPSFStars(self, fitsPath, catalog, check=3, threshold=50000):
        image = fits.getdata(fitsPath)
        skyFlux = np.median(image)
        
        self._debug("Getting PSF stars")
        self._debug("\tEnforcing maximum flux thresholds")
        psfMask = (catalog['FLUX_MAX'] > 5 * skyFlux) & (catalog['FLUX_MAX'] < 0.8 * threshold)  

        self._debug("\tEnforcing CLASS_STAR")
        psfMask = psfMask & (catalog['CLASS_STAR'] > 0.8)
        
        self._debug("\tEnforcing ELLIPTICITY")
        psfMask = psfMask & (catalog['ELLIPTICITY'] < 0.1)
        
        self._debug("\tEnforcing minimum distance from each source")
        for i,row in enumerate(catalog):
            if psfMask[i]:
                dists = np.sqrt((catalog['X_IMAGE'] - row['X_IMAGE'])**2 + (catalog['Y_IMAGE'] - row['Y_IMAGE'])**2)
                minDist = np.min(dists[dists > 1e-6])
                psfMask[i] = minDist > (row['FWHM_IMAGE'] * 3 + 15)
                
        
        self._debug("\tEnforcing no overflow")
        for i,row in enumerate(catalog):
            if psfMask[i]:
                x = np.round(row['X_IMAGE']).astype(np.int)
                y = np.round(row['Y_IMAGE']).astype(np.int)
                psfMask[i] = psfMask[i] and np.all(image[y-check:y+check, x-check:x+check] < threshold)
                psfMask[i] = psfMask[i] and np.all(image[y-1:y+1, x-1:x+1] > 4 * skyFlux)
                psfMask[i] = psfMask[i] and np.all(image[y-check:y+check, x-check:x+check] > 0)
        
        psfMasks = self.splitCatalog(image, catalog)
        for m in psfMasks:
            m &= psfMask
        
                
        self._debug("\tReturning mask for %d,%d candidate masks" % (psfMasks[0].sum(), psfMasks[1].sum()))
        return psfMasks
        
    def getPSFs(self, fitsPath, catalog):
        self._debug("Generating PSFs")
        
        
        if not self.redo:
            pattern = re.compile("^psf[0-9]\.fits$")
            generated = [os.path.abspath(self.outDir + os.sep + f) for f in os.listdir(self.outDir) if pattern.match(f) is not None]
            if len(generated) > 0:
                self._debug("Found existing PSF files: %s"%generated)
                return generated
                
        psfMasks = self.getPSFStars(fitsPath, catalog)
        
        # Copy image
        self._debug("\tCopying image fits to temp directory")
        shutil.copy(fitsPath, self.tempDir + os.sep + "imgPsf.fits")
    
        # Save psf locations
        self._debug("\tCreating PSF star position lists")
        for i,psf in enumerate(psfMasks):
            np.savetxt(self.tempDir + os.sep + "%d.cat"%i, catalog[psf][['X_IMAGE', 'Y_IMAGE']], fmt="%0.2f")
        # Update environ
        env = os.environ.copy()
        env['PATH'] = ("%s/variants/common/bin:%s/bin:%s/python/bin:" % (urekaPath, urekaPath, urekaPath))+env['PATH']
        env['PYTHONEXECUTABLE'] = "%s/python/bin/python" % urekaPath
        # Save script
        script = '''cd %s
    daophot
    digiphot
    phot imgPsf %s output=default scale=1 fwhmpsf=3  sigma=3.5 readnoi=10 gain=GAIN calgori=centroid cbox=5 salgori=mode annulus=25 dannulus=30  aperture=5,8,10,15 zmag=25 interactive=no verify-
    pstselect imgPsf photfile=default pstfile=default maxnpsf=300 psfrad=17 fitrad=3 verify-
    psf imgPsf photfile=default pstfile=default psfimage=default opstfile=default groupfile=default scale=1 fwhmpsf=3  sigma=3.5 readnoi=10 gain=GAIN function=gauss varorder=1 saturate=no nclean=0 psfrad=17 fitrad=3 interac=no verify-
    task als2bl = %s
    als2bl imgPsf.psf.1.fits %s
    
    .exit''' % (os.path.abspath(self.tempDir), "%s", baolabScript, "%s")
    
    
        # Run script for each psf
        psfNames = ["psf%d.fits"%i for i in range(len(psfMasks))]
        
        self._debug("\tSaving script and executing Pyraf")
        for i,name in enumerate(psfNames):
            filename = self.tempDir + os.sep + "script.cl"
            with open(filename, 'w') as f:
                script2 = script % ("%d.cat"%i, name)
                f.write(script2)
            self._debug("\tExecuting pyraf commands for psf catalog %d"%i)
                
            # Run script
            p = subprocess.Popen(["/bin/bash", "-i", "-c", "pyraf -x -s < script.cl && cp %s .."%name], env=env, stderr=subprocess.PIPE, stdout=subprocess.PIPE, cwd=os.path.abspath(self.tempDir))   
            output = p.communicate() #now wait
            #self._debug(output[0])
            #self._debug(output[1])
            
        resultNames = []
        for n in psfNames:
            name = self.tempDir + os.sep + n
            if not os.path.isfile(name):
                raise "Expected file %s not found" % name
            else:
                fullName = os.path.abspath(self.outDir + os.sep + n)
                shutil.copy(name, fullName)
                resultNames.append(fullName)
            
        
        return resultNames
        
    def splitCatalog(self, image, catalog):
        ysplit = image.shape[0] / 2;
        psfMask1 = (catalog['Y_IMAGE'] > ysplit)
        psfMask2 = (catalog['Y_IMAGE'] < ysplit)
        return [psfMask1, psfMask2]
                
        
    def runIshape(self, fitsPath, catalog, gcsMask):
        self._debug("Running ishape on given image")
        tempDir = self.tempDir + os.sep + "ishape"
        self._debug("\tGenerating temp directory at %s" % tempDir)
        
        models = ["KING30"]#, "SERSICx"]
        indexes = [30.0]#, 2.0]
        
        
        try:
            shutil.rmtree(tempDir)  # delete directory
        except OSError as exc:
            if exc.errno != errno.ENOENT:
                raise  # re-raise exception
        os.mkdir(tempDir)
        
    
        splitMasks = self.splitCatalog(fits.getdata(fitsPath), catalog)
        psfs = self.getPSFs(fitsPath, catalog)
    
        tempCoord = tempDir + os.sep + "coords%d.txt"
        for i,splt in enumerate(splitMasks):
            np.savetxt(tempCoord%i, catalog[splt & gcsMask][['X_IMAGE', 'Y_IMAGE']], fmt="%d")        
    
        for i,psf in enumerate(psfs):
            for model,index in zip(models, indexes):
                argumentFile = tempDir + os.sep + "command.bl"
                with open(argumentFile, 'w') as f:
                    f.write("cd %s\nishape %s %s %s LOGMODE=1 SHAPE=%s INDEX=%0.1f FITRAD=7 CALCERR=no\n" % (os.path.abspath(tempDir), os.path.abspath(fitsPath), "coords%d.txt"%i, os.path.abspath(psf), model, index))
                
                commandline = 'bl < %s' % argumentFile
                
                #commandline = "bl < /Users/shinton/Downloads/bl/test.bl"
                self._debug("\tExecuting baolab and ishape for psf %s and model %s (%0.1f)" % (psf, model, index))
                p = subprocess.Popen(["/bin/bash", "-i", "-c", commandline], stderr=subprocess.PIPE, stdout=subprocess.PIPE)
                output = p.communicate() #now wait
                #print(output[0])
                #print(output[1])
        
        self._debug("\tReading log file")
        raw = []            
        with open(tempDir + os.sep + "ishape.log") as log:
            for line in log:
                if "@@F" in line or "@@E" in line:
                    raw.append(line)
        parsed = {}
        # @#F <image>       <x> <y>   <psf>      <SHAPE> <FWHM> <RATIO> <PA> <CHISQR> <CHISQR0> <FLUX> <S/N> [INDEX] 
        # @#E <+/-FWHM> <+/-RATIO> <+/-PA> [+/-INDEX]
        x = ""
        y = ""
        shape = ""
        for r in raw:
            sep = r.split()
            if sep[0] == "@@F" and sep[6] != "OBJ-ERROR" and sep[6] != "FIT-ERROR":
                x = sep[2]
                y = sep[3]
                shape = sep[5]
                fwhm = float(sep[6])
                ratio = float(sep[7])
                pa = float(sep[8])
                chi2 = float(sep[9])
                chi2d = float(sep[10])
                sn = float(sep[12])
                index = sep[13] if len(sep) >= 14 else ""
                shape += index
                key = x + " " + y
                if parsed.get(key) is None:
                    parsed[key] = {}
                parsed[key][shape] = [fwhm, ratio, pa, chi2, chi2d, sn]
            if sep[0] == "@@E":
                parsed[key][shape] += [float(sep[1]), float(sep[2]), float(sep[3]), float(sep[4]), float(sep[5]), float(sep[6])]
        
        return parsed
        
    def getMatch(self, row, parsed):
        x = row['X_IMAGE']
        y = row['Y_IMAGE']
        key = "%d %d" % (x,y)
        #print(key)
        if parsed.get(key) is not None:
            return parsed[key]
        else:
            for i in range(-2,3):
                for j in range(-2,3):
                    key = "%d %d" % (x+i,y+j)
                    if parsed.get(key) is not None:
                        return parsed[key]
        return None
        
    def getRAandDec(self, fitsFile, catalog):
        self._debug("\tRunning xy2sky")
        tempDir = self.tempDir + os.sep + "xy2sky"
        self._debug("\tGenerating temp directory at %s" % tempDir)
        
        try:
            shutil.rmtree(tempDir)  # delete directory
        except OSError as exc:
            if exc.errno != errno.ENOENT:
                raise  # re-raise exception
        os.mkdir(tempDir)
        
        imfile = tempDir + os.sep + "imfile.txt"
        np.savetxt(imfile, catalog[['X_IMAGE','Y_IMAGE']], fmt="%0.3f")
        outfile = tempDir + os.sep + "skys.txt"
        # | cut -d \" \" -f 1-2 > %s
        commandline = wcsToolsPath + "/xy2sky -d %s @%s | awk '{print $1,$2}'> %s" % (fitsFile, imfile, outfile)
        p = subprocess.Popen(["/bin/bash", "-i", "-c", commandline], stderr=subprocess.PIPE, stdout=subprocess.PIPE)
        output = p.communicate() #now wait
        
        res = np.loadtxt(outfile)
        catalog = append_fields(catalog, 'RA', res[:,0], usemask=False)    
        catalog = append_fields(catalog, 'DEC', res[:,1], usemask=False)    
        
        return catalog