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
import tempfile
import shutil
import errno
from matplotlib.ticker import NullFormatter
from sklearn import manifold
from sklearn.utils import check_random_state
from sklearn.cluster import DBSCAN
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn import metrics
from itertools import cycle
from numpy.lib.recfunctions import append_fields
from scipy.interpolate import interp1d
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import subprocess
import stat



from astroquery.irsa_dust import IrsaDust
import pickle
from scipy import interpolate


urekaPath = "/Users/shinton/Software/Ureka"
wcsToolsPath = "/Users/shinton/Software/wcstools/bin"
baolabScript = '/Users/shinton/Software/baolab-0.94.1e/als2bl.cl'
sexPath = '/Users/shinton/Software/Ureka/bin/sex'
scampPath = '/usr/local/bin/scamp'
missfitsPath = '/usr/local/bin/missfits' 


class Dust(object):
    def __init__(self, tempParentDir=None, debug=True, debugPlot=False, tempSubDir="dust"):
        self.tempParentDir = tempParentDir
        self.tempSubDir = tempSubDir
        self.debug = debug
        self.debugPlot = debugPlot
        self.tempDir = None
        self.outDir = "../out" + os.sep + tempSubDir
        outPath = self.outDir + os.sep + "downloaded.pkl"
        
        self._setupTempDir()
        
        if os.path.exists(outPath):
            with open(outPath, 'rb') as f:
                self.image_list = pickle.load(f)
        else:
            with open(outPath, 'wb') as outfile:
                self.image_list = IrsaDust.get_images("maffei1")
                pickle.dump(self.image_list, outfile)

        self.image_types = [i[0].header['BUNIT'] for i in self.image_list]

        self.ebvIndex = self.image_types.index("mag E(B-V)")
        # Schlafly, E.F. & Finkbeiner, D.P.  2011, ApJ 737, 103 (S and F).
        self.extinctions = { "I": 1.698, "R": 2.285, "Z": 1.263 }
        
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
        
    def getExtinctionFits(self):
        return self.image_list[self.ebvIndex]
                
    def getExtinctionImage(self):
        return self.getExtinctionFits()[0].data
        
    def getExtinctions(self, ra, dec, bands=['z','i','r']):
        
        self._debug("\tGetting extinction for given objects")
        
        
        ext = self.getExtinctionImage()
        xs = np.arange(ext.shape[0])
        ys = np.arange(ext.shape[1])
        xx, yy = np.meshgrid(xs, ys)
        f = interpolate.interp2d(xs, ys, ext, kind='cubic')        
        
        self._debug("\tRunning sky2xy")
        tempDir = self.tempDir + os.sep + "sky2xy"
        self._debug("\tGenerating temp directory at %s" % tempDir)
        
        try:
            shutil.rmtree(tempDir)  # delete directory
        except OSError as exc:
            if exc.errno != errno.ENOENT:
                raise  # re-raise exception
        os.mkdir(tempDir)
        
        imfile = tempDir + os.sep + "imfile.txt"
        np.savetxt(imfile, np.vstack((ra,dec)).T, fmt="%0.6f")
        outfile = tempDir + os.sep + "skys.txt"
        
        fitsFile = tempDir + os.sep + "dust.fits"
        self.getExtinctionFits().writeto(fitsFile, clobber=True)
        #
        commandline = wcsToolsPath + "/sky2xy %s @%s  | awk '{print $5,$6}' > %s" % (fitsFile, imfile, outfile)
        p = subprocess.Popen(["/bin/bash", "-i", "-c", commandline], stderr=subprocess.PIPE, stdout=subprocess.PIPE)
        output = p.communicate() #now wait
        self._debug("\tLoading sky2xy results and interpolating")
        res = np.loadtxt(outfile)
        x = res[:,0]    
        y = res[:,1]
        
        ebvs = np.array([f(x[i],y[i]) for i in range(x.size)]).flatten()
        result = np.array([ebvs * self.extinctions[band] for band in bands]).T
        return result
        
    def correctExtinctions(self, cat):
        self._debug("Correcting input catalog")
        bands = ["Z","I","R"]
        ext = self.getExtinctions(cat['RA'], cat['DEC'], bands=bands)
        names = cat.dtype.names
        for i,b in enumerate(bands):
            colsToFix = [n for n in names if n.find("%s_"%b) == 0 and n.find("MAGE") == -1 and n.find("MASK") == -1 ]
            self._debug("\tFixing %s"%colsToFix)
            for col in colsToFix:
                cat[col] -= ext[:,i]
        return cat