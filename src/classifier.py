import os
import sys
import numpy as np
import subprocess

import matplotlib, matplotlib.pyplot as plt
import sextractor
from astropy.io import fits
from scipy.ndimage.filters import *
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from numpy.lib.recfunctions import append_fields
from sklearn.externals import joblib

from reducer import Reducer
from smartClassifier import *

class Classifier(Reducer):
    def __init__(self, fitsFile, candidates, debug=True, debugPlot=False, tempParentDir=None):
        self.fitsFile = fitsFile
        self.candidates = candidates
        self.classifier = None
        super(Classifier, self).__init__(debug=debug, debugPlot=debugPlot, tempParentDir=tempParentDir, tempSubDir="classifier")
       
        
    def getClassifiers(self):
        
        class1Save = self.outDir + os.sep + "classifier1.pkl'
        class2Save = self.outDir + os.sep + "classifier2.pkl'
        
        class1Exists = os.path.exists(class1Save)
        class2Exists = os.path.exists(class2Save)
        
        if not (class1Exists and class2Exists):
            self._setupTempDir()
            self.catalog = self.getCatalog(self.fitsFile)
            self.candidateMask = self._getCandidateMask(self.catalog, np.loadtxt(self.candidates))
        if class1Exists:
            self.classifier = joblib.load(class1Save)
        else:
            self.classifier,catalog,candidateMask = self._trainClassifier()
            joblib.dump(self.classifier, class1Save) 

        if class2Exists:
            self.classifier2 = joblib.load(class2Save)
        else:
            self.classifier2 = self._trainClassifier2(catalog, candidateMask)
            joblib.dump(self.classifier2, class2Save) 

        #self._testClassifier(catalog, candidateMask)
        #self._cleanTempDir()
        self._debug("Classifier generated. Now you can invoke .clasify(catalog)")

    def _getCandidateMask(self, catalog, coord, pixelThreshold=4):
        self._debug("Loading in extendeds")
        mask = []
        for entry in catalog:
            dists = np.sqrt((entry['X_IMAGE'] - coord[:,0])**2 + (entry['Y_IMAGE'] - coord[:,1])**2 )
            mask.append(dists.min() < pixelThreshold)
        mask = np.array(mask)
        return mask 
        
    def _getArtificial(self, i):
        self._debug("Generating artifical populations for training")
        tempDir = self.tempDir + os.sep + "artificial"
        outDir = self.outDir + os.sep + "artificial"
        if not os.path.exists(tempDir):
            os.mkdir(tempDir)
        if not os.path.exists(outDir):
            os.mkdir(outDir)
        psfs = self.getPSFs(self.fitsFile, self.catalog)
        image = fits.getdata(self.fitsFile)
        
        kingPSFs = [self._generateKingPSF(tempDir, outDir, psf) for psf in psfs]

            
        return self._getArtificialImage(i, tempDir, outDir, psfs, kingPSFs, image)
            
        
            
                
    def _getArtificialImage(self, i, tempDir, outDir, psfs, kingPSFs, image, numExtended=300, numStars=3000):
        starListFile = outDir + os.sep + "startList%d.txt" % i
        extendedListFile = outDir + os.sep + "extendedList%d.txt" % i
        imageFile = outDir + os.sep + "artificial%d.fits" % i
        starListFileT = tempDir + os.sep + "startList%d.txt" % i
        extendedListFileT = tempDir + os.sep + "extendedList%d.txt" % i
        imageFileT = tempDir + os.sep + "artificial%d.fits" % i
        
        
        if not self.redo and os.path.exists(starListFile) and os.path.exists(extendedListFile) and os.path.exists(imageFile):
            return (np.loadtxt(starListFile), np.loadtxt(extendedListFile), fits.getdata(imageFile))
        
        x = image.shape[1]
        yy = image.shape[0]/len(psfs)
        if (yy % 2 == 0):
            ys = [yy,yy]
        else:
            ys = [yy, yy+1]
        
        
        starList = None
        extendedList = None
        finImage = None
        for kingPSFList, psf, y in zip(kingPSFs, psfs, ys):
            starListX = np.random.uniform(low=0, high=x, size=numStars/len(psfs))
            starListY = np.random.uniform(low=0, high=y, size=numStars/len(psfs))
            starListMag = np.random.uniform(low=11, high=16, size=numStars/len(psfs))
            starListTemp = np.vstack((starListX, starListY, starListMag)).T
            extendedListX = np.random.uniform(low=0, high=x, size=numExtended/len(psfs))
            extendedListY = np.random.uniform(low=0, high=y, size=numExtended/len(psfs))
            extendedListMag = np.random.uniform(low=11, high=16, size=numExtended/len(psfs))
            extendedListTemp = np.vstack((extendedListX, extendedListY, extendedListMag)).T
        
            np.savetxt(starListFileT, starListTemp, fmt="%0.2f")
            np.savetxt(extendedListFileT, extendedListTemp, fmt="%0.2f")
            
            argumentFile = tempDir + os.sep + "command.bl"
            commandline = 'bl < %s' % argumentFile
            
            with open(argumentFile, 'w') as f:
                f.write("cd %s\nmksynth %s %s PSFTYPE=USER PSFFILE=%s REZX=%d REZY=%d ZPOINT=%s BACKGR=0 RDNOISE=0\n" % (os.path.abspath(tempDir), os.path.abspath(starListFileT), os.path.abspath(imageFileT), os.path.abspath(psf), x, y, "1000000000"))
            self._debug("\tGenerating stars for artificial image %d"%i)
            p = subprocess.Popen(["/bin/bash", "-i", "-c", commandline], stderr=subprocess.PIPE, stdout=subprocess.PIPE)
            output = p.communicate()
            print(output[0])
            print(output[1])
            
            img = fits.getdata(imageFileT)
            #img -= np.median(img)
            
            for i,kingPSF in enumerate(kingPSFList):
                extendedListSpecific = extendedListTemp[i::len(kingPSFList),:]
                np.savetxt(extendedListFileT, extendedListSpecific, fmt="%0.2f")

                print(kingPSF)
                with open(argumentFile, 'w') as f:
                    f.write("cd %s\nmksynth %s %s PSFTYPE=USER PSFFILE=%s REZX=%d REZY=%d ZPOINT=%s BACKGR=0 RDNOISE=0\n" % (os.path.abspath(tempDir), os.path.abspath(extendedListFileT), os.path.abspath(imageFileT), os.path.abspath(kingPSF), x, y, "1000000000"))
                self._debug("\tGenerating extendeds for artificial image %d"%i)
                p = subprocess.Popen(["/bin/bash", "-i", "-c", commandline], stderr=subprocess.PIPE, stdout=subprocess.PIPE)
                output = p.communicate()
                print(output[0])
                print(output[1])
                
                extendedImage = fits.getdata(imageFileT)
                #extendedImage -= np.median(extendedImage)
                img += extendedImage
            #img = extendedImage
            if starList is None:
                starList = starListTemp
            else:
                starListTemp[:,1] += yy
                starList = np.vstack((starListTemp, starList))
            if extendedList is None:
                extendedList = extendedListTemp
            else:
                extendedListTemp[:,1] += yy
                extendedList = np.vstack((extendedListTemp, extendedList))
            if finImage is None:
                finImage = img
            else:
                finImage = np.vstack((finImage, img))
             
        np.savetxt(starListFile, starList, fmt="%0.2f")
        np.savetxt(extendedListFile, extendedList, fmt="%0.2f")
        '''
        fig = plt.figure(figsize=(20,7))
        ax0 = fig.add_subplot(1,1,1)
        ax0.imshow(finImage, origin='upper', cmap='viridis')
        plt.show()
        '''
        fits.writeto(imageFile, finImage)
        
        
        return (starList, extendedList, finImage)
    
    
    
    def _generateKingPSF(self, tempDir, outDir, psf, numKings=20, minFWHM=1.3, std=0.5):
        np.random.seed(111)
        vals = np.abs(np.random.normal(scale=std, size=(numKings))) + minFWHM
        kings = []
        for i,val in enumerate(vals):        
            kingPSF = outDir + os.sep + "king_%d_"%i + os.path.basename(psf)
            if not self.redo and os.path.exists(kingPSF):
                kings.append(kingPSF)
                continue
            argumentFile = tempDir + os.sep + "command.bl"
            with open(argumentFile, 'w') as f:
                f.write("cd %s\nmkcmppsf %s PSFFILE=%s OBJTYPE=KINGx RADIUS=17 INDEX=30 FWHMOBJX=%0.2f FWHMOBJY=%0.2f \n" % (tempDir, os.path.abspath(kingPSF), os.path.abspath(psf), val, val))
            
            commandline = 'bl < %s' % argumentFile
            
            #commandline = "bl < /Users/shinton/Downloads/bl/test.bl"
            self._debug("\tExecuting baolab and mkcmppsf for psf %s" % (os.path.abspath(psf)))
            p = subprocess.Popen(["/bin/bash", "-i", "-c", commandline], stderr=subprocess.PIPE, stdout=subprocess.PIPE)
            output = p.communicate() #now wait
            #print(output[0])
            #print(output[1])
            kings.append(kingPSF)
        return kings

        
    def _getTrainingData(self, numImages=5):
        tempDir = self.tempDir + os.sep + "data"
        if not os.path.exists(tempDir):
            os.mkdir(tempDir)
        
        baseImage = fits.getdata(self.fitsFile)
        catalog = None
        candidateMask = None
        cand = np.loadtxt(self.candidates)
        for i in range(numImages):
            fitsFile = tempDir + os.sep + "artificalAdded%d.fits"%i
            stars, extendeds, image = self._getArtificial(i)
            hdulist = fits.open(self.fitsFile)
            hdulist[0].data = baseImage + image
            #hdulist[0].data = image
            hdulist.writeto(fitsFile, clobber=True)
            hdulist.close()
            catalogTemp = self.getCatalog(fitsFile)
            if catalog is None:
                catalog = catalogTemp
            else:
                catalog = np.concatenate((catalog, catalogTemp))
                
            extendeds = np.vstack((extendeds[:,:-1], cand))
            #extendeds = extendeds[:,:-1]
            candidateMaskTemp = self._getCandidateMask(catalogTemp, extendeds)
            if candidateMask is None:
                candidateMask = candidateMaskTemp
            else:
                candidateMask = np.concatenate((candidateMask, candidateMaskTemp))
            
        return catalog, candidateMask
        
    def _getTrainingData2(self, numImages=3):
        tempDir = self.tempDir + os.sep + "data"
        if not os.path.exists(tempDir):
            os.mkdir(tempDir)
        
        baseImage = fits.getdata(self.fitsFile)
        catalog = None
        candidateMask = None
        cand = np.loadtxt(self.candidates)
        for i in range(numImages):
            fitsFile = tempDir + os.sep + "artificalAdded%d.fits"%i
            stars, extendeds, image = self._getArtificial(i)
            hdulist = fits.open(self.fitsFile)
            hdulist[0].data = baseImage + image
            #hdulist[0].data = image
            hdulist.writeto(fitsFile, clobber=True)
            hdulist.close()
            catalogTemp = self.getCatalog(fitsFile, ishape=True)
            
            

            if catalog is None:
                catalog = catalogTemp
            else:
                catalog = np.concatenate((catalog, catalogTemp))
                
            extendeds = np.vstack((extendeds[:,:-1], cand))
            #extendeds = extendeds[:,:-1]
            candidateMaskTemp = self._getCandidateMask(catalogTemp, extendeds)
            if candidateMask is None:
                candidateMask = candidateMaskTemp
            else:
                candidateMask = np.concatenate((candidateMask, candidateMaskTemp))
                
        return catalog, candidateMask
                
    
        
    def _trainClassifier(self):
        
        catalog, candidateMask = self._getTrainingData(numImages=5)
        self.catalog1 = catalog
        self.mask1 = candidateMask
        self._debug("Have data for %d extendeds out of %d objects (%0.2f%%)" % (candidateMask.sum(), candidateMask.size, 100.0*candidateMask.sum()/candidateMask.size))
        y = candidateMask * 1
        X = catalog.view(np.float64).reshape(catalog.shape + (-1,))[:, 3:]
        
        
        self._debug("Creating classifier")

        # Create and fit an AdaBoosted decision tree
        self.sc = SmartClassifier(X,y)
        bdt = self.sc.learn()
        #bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), algorithm="SAMME", n_estimators=strength)
        #bdt.fit(X, y, sample_weight=1+40*y)
        self._debug("Classifier created")
        bdt.label = "Boosted Decision Tree"
        return bdt,catalog, candidateMask
    
    
    def classify(self, catalog, ellipticity=0.25):
        X = catalog.view(np.float64).reshape(catalog.shape + (-1,))[:, 3:]
        z = self.classifier.predict(X) == 1
        gcs = z & (catalog['ELLIPTICITY'] < ellipticity)
        return gcs
        
    def classify2(self, catalog, ellipticity=0.25):
        X = catalog.view(np.float64).reshape(catalog.shape + (-1,))[:, 3:]
        z = self.classifier2.predict(X) == 1
        gcs = z & (catalog['ELLIPTICITY'] < ellipticity)
        return gcs
    
    def _trainClassifier2(self, catalog, catalogMask):
        catalog, candidateMask = self._getTrainingData2(numImages=5)
        self.catalog2 = catalog
        self.mask2 = candidateMask
        self._debug("Have data for %d extendeds out of %d objects (%0.2f%%)" % (candidateMask.sum(), candidateMask.size, 100.0*candidateMask.sum()/candidateMask.size))
        y = candidateMask * 1
        X = catalog.view(np.float64).reshape(catalog.shape + (-1,))[:, 3:]
        
        
        self._debug("Creating classifier")

        # Create and fit an AdaBoosted decision tree
        self.sc2 = SmartClassifier(X,y)
        bdt2 = self.sc2.learn()
        #bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), algorithm="SAMME", n_estimators=strength)
        #bdt.fit(X, y, sample_weight=1+40*y)
        self._debug("Classifier created")
        bdt2.label = "Boosted Decision Tree"
        return bdt2

        