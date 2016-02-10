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
        self.catalog = self.getCatalog(self.fitsPath)
        self.gcMask = self.classifier.classify(catalog)
        ishapeRes = self.runIshape(self.fitsPath, catalog, self.gcMask)
        
        self.catalogFinal = self.addIshapeDetails(catalog[gcMask], ishapeRes)        
        self.gcMask2 = self.classifier.classify2(self.catalogFinal)
        
        self.gcsCatalog = self.catalogFinal[self.gcMask2]
        self.gcsCatalog = self.getRAandDec(self.fitsPath, self.gcsCatalog)
        np.save(outFile, self.gcsCatalog)
        return self.gcsCatalog
        

                
        

        
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
        

        

        
            
    
        