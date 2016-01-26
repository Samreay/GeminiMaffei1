import os
import sys
import tempfile
import shutil
import errno
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
        
        self.catalogFinal = self.addIshapeDetails(catalog[gcMask], ishapeRes)        
        gcMask2 = self.classifier.classify2(self.catalogFinal)
        
        self.gcsCatalog = self.catalogFinal[gcMask2]
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
        

        

        
            
    
        