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
from helper import *
from reducer import *

class Mosaic(Reducer):
    def __init__(self, fitsPath, debug=True, debugPlot=False, tempParentDir=None):
        self.fitsPath = os.path.abspath(fitsPath)
        self.subName = os.path.splitext(os.path.basename(self.fitsPath))[0]
        super(Mosaic, self).__init__(debug=debug, debugPlot=debugPlot, tempParentDir=tempParentDir, tempSubDir="mosaic_" + self.subName)
        if not self.tempDirSetup: 
            self._setupTempDir()
            
    def importCatalog(self, catalog):
        catalog = catalog.copy()
        catalog = self.updatePixelPositions(self.fitsPath, catalog)
        self.catalog = catalog
        
    def updatePixelPositions(self, fitsFile, catalog):
        self._debug("\tRunning xy2sky")
        tempDir = self.tempDir + os.sep + "sky2xy"
        self._debug("\tGenerating temp directory at %s" % tempDir)
        
        try:
            shutil.rmtree(tempDir)  # delete directory
        except OSError as exc:
            if exc.errno != errno.ENOENT:
                raise  # re-raise exception
        os.mkdir(tempDir)
        
        imfile = tempDir + os.sep + "imfile.txt"
        np.savetxt(imfile, catalog[['RA','DEC']], fmt="%0.6f")
        outfile = tempDir + os.sep + "skys.txt"
        #
        commandline = wcsToolsPath + "/sky2xy %s @%s  | awk '{print $5,$6}' > %s" % (fitsFile, imfile, outfile)
        p = subprocess.Popen(["/bin/bash", "-i", "-c", commandline], stderr=subprocess.PIPE, stdout=subprocess.PIPE)
        output = p.communicate() #now wait
        
        res = np.loadtxt(outfile)
        catalog['X_IMAGE'] =  res[:,0]    
        catalog['Y_IMAGE'] =  res[:,1]    
        
        return catalog
        
    def show(self):
        showInDS9(self.fitsPath, catalog=self.catalog)
                