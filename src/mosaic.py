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
                
    def getMagnitudes(self, threshold=0.0004):
        outFile = self.outDir + os.sep + "cat" + self.subName + ".npy"
        if os.path.exists(outFile):
            cat = np.load(outFile)
        else:
            cat,sex = self._getCatalogs(self.fitsPath, None, aperture=True)
            cat = self.getRAandDec(self.fitsPath, cat)
            np.save(outFile, cat)
        mag = []
        magE = []
        for entry in self.catalog:
            ra = entry['RA']
            dec = entry['DEC']
            dists = np.sqrt((cat['RA']-ra)*(cat['RA']-ra) + (cat['DEC']-dec)*(cat['DEC']-dec))
            minDist = dists.argmin()
            if dists[minDist] < threshold:
                entry['RA'] = ra
                entry['DEC'] = dec
                mag.append(cat[minDist]['MAG_BEST'])
                magE.append(cat[minDist]['MAGERR_BEST'])
            else:
                mag.append(np.NaN)
                magE.append(np.NaN)
        return np.array(mag), np.array(magE)
        
    def runPhot(self, fitsFile, cat, apertures=[1,2,3,4,5,6,7,8,9,10,11]):
        outfile = self.outDir + os.sep + "outfile_%d.txt"%cat.shape[0]
        if os.path.exists(outfile):
            mags = np.loadtxt(outfile)
        else:
            self._debug("\tRunning phot")
            tempDir = self.tempDir + os.sep + "phot"
            self._debug("\tGenerating temp directory at %s" % tempDir)
            
            try:
                shutil.rmtree(tempDir)  # delete directory
            except OSError as exc:
                if exc.errno != errno.ENOENT:
                    raise  # re-raise exception
            os.mkdir(tempDir)
                    
            positions = cat[['X_IMAGE', 'Y_IMAGE']]
            positionFile = tempDir + os.sep + "pos.lst"
            np.savetxt(positionFile, positions, fmt="%0.3f")
            
            env = os.environ.copy()
            env['PATH'] = ("%s/variants/common/bin:%s/bin:%s/python/bin:" % (urekaPath, urekaPath, urekaPath))+env['PATH']
            env['PYTHONEXECUTABLE'] = "%s/python/bin/python" % urekaPath
            # Save script
            script = '''    cd %s
        daophot
        digiphot
        phot %s %s output=out.mag scale=1 fwhmpsf=3 sigma=3.5 readnoi=10 calgori=centroid cbox=5 salgori=mode annulus=25 dannulus=30  apertures=%s zmag=25 interactive=no verify-
        txdump out.mag mag yes > outfile.txt
        .exit''' % (os.path.abspath(tempDir), os.path.abspath(fitsFile), os.path.abspath(positionFile), ",".join(["%d"%i for i in apertures]))
        
            scriptFile = tempDir + os.sep + "script.cl"
            with open(scriptFile, 'w') as f:
                f.write(script)
    
                
            # Run script
            self._debug("\tRunning script")
            p = subprocess.Popen(["/bin/bash", "-i", "-c", "pyraf -x -s < %s && sed 's/INDEF/99/g' outfile.txt > outfile2.txt"%os.path.abspath(scriptFile)], env=env, stderr=subprocess.PIPE, stdout=subprocess.PIPE, cwd=os.path.abspath(tempDir))   
            output = p.communicate() #now wait
            self._debug(output[0])
            self._debug(output[1])
            mags = np.loadtxt(tempDir + os.sep + "outfile2.txt")
            np.savetxt(outfile, mags, fmt="%0.6f")
        
        char = self.subName[self.subName.find("_")+1:]
        for i,a in enumerate(apertures):
            column = "%s_%d"%(char,a)
            cat = append_fields(cat, column, mags[:,i], usemask=False)
        return cat
    
    def getPhot(self, cat):
        print(self.subName)
        self._debug("Running phot to get colours on mosaic %s" % self.subName)
        cat = self.updatePixelPositions(self.fitsPath, cat)
        
        data = self.runPhot(self.fitsPath, cat)
        
        return data
                
                