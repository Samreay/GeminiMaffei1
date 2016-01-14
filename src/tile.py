import os
import sys
import tempfile
import shutil
import errno
import subprocess
import re

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
from reducer import *
from helper import *

urekaPath = "/Users/shinton/Software/Ureka"
wcsToolsPath = "/Users/shinton/Software/wcstools/bin"
baolabScript = '/Users/shinton/Software/baolab-0.94.1e/als2bl.cl'
sexPath = '/Users/shinton/Software/Ureka/bin/sex'
scampPath = '/usr/local/bin/scamp'
missfitsPath = '/usr/local/bin/missfits' 


class Tile(Reducer):
    def __init__(self, fitsPath, classifier, debug=True, debugPlot=False, tempParentDir=None):
        self.classifier = classifier
        self.fitsPath = os.path.abspath(fitsPath)
        self.subName = os.path.splitext(os.path.basename(self.fitsPath))[0]
        super(Tile, self).__init__(debug=debug, debugPlot=debugPlot, tempParentDir=tempParentDir, tempSubDir="tile_" + self.subName)
        
    def getGlobalClusters(self):
        if not self.tempDirSetup: 
            self._setupTempDir()
        outFile = self.outDir + os.sep + self.subName + "GCCatalog.npy"
        if not self.redo and os.path.exists(outFile):
            self._debug("Returning previously generated GCs")
            return np.load(outFile)
        catalog = self.getCatalog(self.fitsPath)
        gcMask = self.classifier.classify(catalog)
        ishapeRes = self.runIshape(self.fitsPath, catalog, gcMask)
        self.gcsCatalog = self.refineGCMask(catalog, gcMask, ishapeRes)
        self.gcsCatalog = self.getRAandDec(self.fitsPath, self.gcsCatalog)
        np.save(outFile, self.gcsCatalog)
        return self.gcsCatalog
        
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
        
    def refineGCMask(self, catalog, mask, parsed):
        
        self._debug("Refining GC mask using ishape results")
        newMask = mask.copy()
        cds = []
        css = []
        kingFwhm = []
        result = []
        for i,row in enumerate(catalog):
            if newMask[i]:        
                p = self.getMatch(row, parsed)
                if p is not None:
                    chi2King = p['KING30'][3]
                    chi2Delta = p['KING30'][4]
                    fwhm = p['KING30'][0]
                    #chi2Sersic2 = p['SERSICx2.00'][3]
                    val = chi2Delta/chi2King
                    newMask[i] &= val > 1.3
                    if newMask[i]:
                        cds.append(np.round(val, decimals=5))
                        #css.append(chi2Sersic2/chi2King)
                        kingFwhm.append(fwhm)
                else:
                    newMask[i] = False
        catalogNew = catalog[newMask]
        catalogNew = append_fields(catalogNew, 'Chi2DeltaKing', cds, usemask=False)    
        #catalogNew = append_fields(catalogNew, 'Chi2SersicKing', css, usemask=False)    
        catalogNew = append_fields(catalogNew, 'KingFWHM', kingFwhm, usemask=False)
        self._debug("\tNew catalog has %d gcs, compared to %d previously" % (newMask.sum(), mask.sum()))
        return catalogNew
        
    def splitCatalog(self, image, catalog):
        ysplit = image.shape[0] / 2;
        psfMask1 = (catalog['Y_IMAGE'] > ysplit)
        psfMask2 = (catalog['Y_IMAGE'] < ysplit)
        return [psfMask1, psfMask2]
        
        
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
            self._debug(output[0])
            self._debug(output[1])
            
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
                    f.write("cd %s\nishape %s %s %s LOGMODE=1 SHAPE=%s INDEX=%0.1f FITRAD=3 CALCERR=no\n" % (tempDir, fitsPath, "coords%d.txt"%i, os.path.abspath(psf), model, index))
                
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
            if sep[0] == "@@F" and sep[6] != "OBJ-ERROR":
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
        