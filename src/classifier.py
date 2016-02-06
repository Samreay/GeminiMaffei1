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
import hashlib
from reducer import Reducer
from smartClassifier import *

class Classifier(Reducer):
    def __init__(self, fitsFolder, debug=True, debugPlot=False, tempParentDir=None):
        self.fitsFolder = fitsFolder
        self.fitsPath = fitsFolder
        self.classifier = None
        self.candidateMask = {}
        self.mainCatalog = {}
        super(Classifier, self).__init__(debug=debug, debugPlot=debugPlot, tempParentDir=tempParentDir, tempSubDir="classifier")
       
        
    def getClassifiers(self):
        if not os.path.exists(self.outDir):
            os.mkdir(self.outDir)
        outDir = self.outDir + os.sep + "classPickle"
        if not os.path.exists(outDir):
            os.mkdir(outDir)
        class1Save = outDir + os.sep + "classifier1.pkl"
        class2Save = outDir + os.sep + "classifier2.pkl"
        
        class1Exists = os.path.exists(class1Save)
        class2Exists = os.path.exists(class2Save)

        if not (class1Exists and class2Exists):
            self._setupTempDir()
            self.fitsFiles = [f[:-5] for f in os.listdir(self.fitsFolder) if ".fits" in f]
            self.fitsFilesLoc = [os.path.abspath(self.fitsFolder + os.sep + f) for f in os.listdir(self.fitsFolder) if ".fits" in f]
            
            for f in self.fitsFiles:
                self.mainCatalog[f] = self.getCatalog(self.fitsFolder + os.sep + f + ".fits", ishape=True)
                self.candidateMask[f] = self._getCandidateMask(self.mainCatalog[f], np.loadtxt(self.fitsFolder + os.sep + f + ".txt"))
                self.mainCatalog[f] = append_fields(self.mainCatalog[f], 'WEIGHT', self.candidateMask[f] * 1.0, usemask=False)    
                self.mainCatalog[f] = append_fields(self.mainCatalog[f], 'EXTENDED', self.candidateMask[f], usemask=False)    
                self.mainCatalog[f] = append_fields(self.mainCatalog[f], 'HLR', np.zeros(self.mainCatalog[f].shape), usemask=False)    
                self.mainCatalog[f] = append_fields(self.mainCatalog[f], 'MAG', np.zeros(self.mainCatalog[f].shape), usemask=False)
            self._trainClassifier()
            joblib.dump(self.sc, class1Save) 
            joblib.dump(self.sc2, class2Save) 
        else:
            self.sc = joblib.load(class1Save)
            self.sc2 = joblib.load(class2Save)
            

        #self._testClassifier(catalog, candidateMask)
        #self._cleanTempDir()
        self._debug("Classifier generated. Now you can invoke .clasify(catalog)")


        
        
    def _getArtificial(self, fitsFile, f, i):
        self._debug("Generating artifical populations for training")
        print("A", fitsFile)
        tempDir = self.tempDir + os.sep + "artificial_%s"%f
        outDir = self.outDir + os.sep + "artificial_%s"%f
        if not os.path.exists(tempDir):
            os.mkdir(tempDir)
        if not os.path.exists(outDir):
            os.mkdir(outDir)
        psfs = self.getPSFs(fitsFile, self.mainCatalog[f])
        image = fits.getdata(fitsFile)
        
        kingPSFs = [self._generateKingPSF(tempDir, outDir, psf,int(int(hashlib.md5(f).hexdigest(), 16) % 1e3)+i) for psf in psfs]

        return self._getArtificialImage(i, tempDir, outDir, psfs, kingPSFs, image)
            
        
            
                
    def _getArtificialImage(self, i, tempDir, outDir, psfs, kingPSFs, image, numExtended=100, numStars=600):
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
        elif image.shape[0] % 2 == 1:
            ys = [yy, yy+1]
        else:
            ys = [yy,yy]
        
        starList = None
        extendedList = None
        finImage = None
        ext = None
        for kingPSFList, psf, y in zip(kingPSFs, psfs, ys):
            starListX = np.random.uniform(low=0, high=x, size=numStars/len(psfs))
            starListY = np.random.uniform(low=0, high=y, size=numStars/len(psfs))
            starListMag = np.random.uniform(low=11, high=17, size=numStars/len(psfs))
            starListTemp = np.vstack((starListX, starListY, starListMag)).T
            extendedListX = np.random.uniform(low=0, high=x, size=numExtended/len(psfs))
            extendedListY = np.random.uniform(low=0, high=y, size=numExtended/len(psfs))
            extendedListMag = np.random.uniform(low=11, high=17, size=numExtended/len(psfs))
            extendedListTemp = np.vstack((extendedListX, extendedListY, extendedListMag)).T
        
            np.savetxt(starListFileT, starListTemp, fmt="%0.3f")
            np.savetxt(extendedListFileT, extendedListTemp, fmt="%0.3f")
            
            argumentFile = tempDir + os.sep + "command.bl"
            commandline = 'bl < %s' % argumentFile
            
            with open(argumentFile, 'w') as f:
                f.write("cd %s\nmksynth %s %s PSFTYPE=USER PSFFILE=%s REZX=%d REZY=%d ZPOINT=%s BACKGR=0 RDNOISE=0\n" % (os.path.abspath(tempDir), os.path.abspath(starListFileT), os.path.abspath(imageFileT), os.path.abspath(psf), x, y, "1000000000"))
            self._debug("\tGenerating stars for artificial image %d"%i)
            p = subprocess.Popen(["/bin/bash", "-i", "-c", commandline], stderr=subprocess.PIPE, stdout=subprocess.PIPE)
            output = p.communicate()
            self._debug("Adding %d stars"%starListTemp.shape[0])
            #print(output[0])
            #print(output[1])
            
            img = fits.getdata(imageFileT)
            #img -= np.median(img)
            extt = None
            for i,kingPSF in enumerate(kingPSFList):
                extendedListSpecific = extendedListTemp[i::len(kingPSFList),:]
                np.savetxt(extendedListFileT, extendedListSpecific, fmt="%0.2f")

                print(kingPSF)
                with open(argumentFile, 'w') as f:
                    f.write("cd %s\nmksynth %s %s PSFTYPE=USER PSFFILE=%s REZX=%d REZY=%d ZPOINT=%s BACKGR=0 RDNOISE=0\n" % (os.path.abspath(tempDir), os.path.abspath(extendedListFileT), os.path.abspath(imageFileT), os.path.abspath(kingPSF[1]), x, y, "1000000000"))
                self._debug("\tGenerating extendeds for artificial image %d"%i)
                p = subprocess.Popen(["/bin/bash", "-i", "-c", commandline], stderr=subprocess.PIPE, stdout=subprocess.PIPE)
                output = p.communicate()
                #print(output[0])
                #print(output[1])
                
                extendedImage = fits.getdata(imageFileT)
                self._debug("Adding KING: radius %0.3f, %d stars"%(kingPSF[0], extendedListSpecific.shape[0]))
                extendedListSpecific = np.hstack((extendedListSpecific, kingPSF[0] * np.ones((extendedListSpecific.shape[0], 1))))
                if extt is None:
                    extt = extendedListSpecific
                else:
                    extt = np.vstack((extt, extendedListSpecific))
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
            if ext is None:
                ext = extt
            else:
                extt[:,1] += yy
                ext = np.vstack((ext, extt))
             
        np.savetxt(starListFile, starList, fmt="%0.3f")
        np.savetxt(extendedListFile, ext, fmt="%0.3f")
        '''
        fig = plt.figure(figsize=(20,7))
        ax0 = fig.add_subplot(1,1,1)
        ax0.imshow(finImage, origin='upper', cmap='viridis')
        plt.show()
        '''
        fits.writeto(imageFile, finImage)
        
        
        return (starList, ext, finImage)
    
    
    
    def _generateKingPSF(self, tempDir, outDir, psf, i, numKings=10, minFWHM=1, meanFWHM=5, std=2):
        np.random.seed(111+i)
        parsecs = np.abs(np.random.normal(loc=meanFWHM, scale=std, size=(numKings)))
        #print(parsecs)
        parsecs[parsecs < minFWHM] += meanFWHM
        as2Rad = np.pi / (180 * 60 * 60)
        pixels = (0.2 / np.tan(2.7e6 * as2Rad)) * parsecs
        kings = []
        for i,pixel in enumerate(pixels):        
            kingPSF = outDir + os.sep + "king_%d_%0.3f_"%(i,parsecs[i]) + os.path.basename(psf)
            if not self.redo and os.path.exists(kingPSF):
                kings.append((parsecs[i], kingPSF))
                continue
            argumentFile = tempDir + os.sep + "command.bl"
            with open(argumentFile, 'w') as f:
                f.write("cd %s\nmkcmppsf %s PSFFILE=%s OBJTYPE=KINGx RADIUS=17 INDEX=30 FWHMOBJX=%0.3f FWHMOBJY=%0.3f \n" % (tempDir, os.path.abspath(kingPSF), os.path.abspath(psf), pixel, pixel))
            
            commandline = 'bl < %s' % argumentFile
            
            #commandline = "bl < /Users/shinton/Downloads/bl/test.bl"
            self._debug("\tGenerating KingPSF %0.3f pizels: mkcmppsf for psf %s" % (pixel, os.path.abspath(psf)))
            p = subprocess.Popen(["/bin/bash", "-i", "-c", commandline], stderr=subprocess.PIPE, stdout=subprocess.PIPE)
            output = p.communicate() #now wait
            #print(output[0])
            #print(output[1])
            kings.append((parsecs[i], kingPSF))
        return kings

    def getCatAndCandFromImage(self, f,loc, tempDir, index, ishape=True):
        dirr = tempDir + os.sep + f
        if not os.path.exists(dirr):
            os.mkdir(dirr)
        fitsFile = dirr + os.sep + "%s_artificalAdded%d.fits"%(f,index)
        print("B", fitsFile)
        stars, extendeds, image = self._getArtificial(loc, f, index)
        hdulist = fits.open(loc)
        print(hdulist[0].data.shape)
        print(image.shape)
        hdulist[0].data += image
        hdulist[0].data = hdulist[0].data.astype(np.float32)
        hdulist.writeto(fitsFile, clobber=True)
        hdulist.close()
        catalogTemp = self.getCatalog(fitsFile, ishape=ishape, excluded=self.mainCatalog[f])
        hlr = []
        mags = []
        ext = []
        for i,row in enumerate(catalogTemp):
            x = row['X_IMAGE']
            y = row['Y_IMAGE']
            dists = np.sqrt((x-extendeds[:,0])*(x-extendeds[:,0]) + (y-extendeds[:,1])*(y-extendeds[:,1]))
            minDist = dists.argmin()
            if dists[minDist] < 4:
                hlr.append(extendeds[minDist, 3])
                mags.append(extendeds[minDist, 2])
                ext.append(True)
            else:
                hlr.append(0)
                mags.append(0)
                ext.append(False)
        catalogTemp = append_fields(catalogTemp, 'WEIGHT', np.zeros(catalogTemp.shape), usemask=False)    
        catalogTemp = append_fields(catalogTemp, 'EXTENDED', np.array(ext), usemask=False)    
        catalogTemp = append_fields(catalogTemp, 'HLR', np.array(hlr), usemask=False)    
        catalogTemp = append_fields(catalogTemp, 'MAG', np.array(mags), usemask=False)
        return catalogTemp, extendeds
        
    def _getTrainingData(self, numImages=200, ishape=True):
        tempDir = self.tempDir + os.sep + "data"
        outFile1 = self.outDir + os.sep + "allData1_%d.npy"%numImages
        outFile2 = self.outDir + os.sep + "allData2_%d.npy"%numImages
        if os.path.exists(outFile1) and os.path.exists(outFile2):
            self._debug("Loading existing catalogs")
            return np.load(outFile1), np.load(outFile2)
            
        if not os.path.exists(tempDir):
            os.mkdir(tempDir)
        
        catalog = np.concatenate(self.mainCatalog.values())
        exts = None
        for i in range(numImages):
            for fitsFile, fitsFileLoc in zip(self.fitsFiles, self.fitsFilesLoc):
                print(fitsFile)
                cat, ext = self.getCatAndCandFromImage(fitsFile, fitsFileLoc, tempDir, i, ishape=ishape)
                catalog = np.concatenate((catalog, cat))
                if exts is None:
                    exts = ext
                else:
                    exts = np.vstack((exts, ext))
        self._debug("Saving catalog to %s"%outFile1)
        np.save(outFile1, catalog)
        np.save(outFile2, exts)
        return catalog, exts
        
    
        
    def _trainClassifier(self, ishape=True):
        
        catalog, extendeds = self._getTrainingData(numImages=1, ishape=ishape)
        self.catalog1 = catalog
        self._debug("Have data for %d extendeds out of %d objects (%0.2f%%)" % (catalog['EXTENDED'].sum(), catalog.shape[0], 100.0*catalog['EXTENDED'].sum()/catalog.shape[0]))
        y = 'EXTENDED'
        
        
        self._debug("Creating classifier")

        # Create and fit an AdaBoosted decision tree
        self.sc = SmartClassifier("1",catalog,y,extendeds,remove=['NUMBER','X_IMAGE','Y_IMAGE','WEIGHT','HLR','MAG','Chi2DeltaKingDiv','Chi2DeltaKingSub','Chi2King','Chi2Delta','KingFWHM'])
        self.classifier = self.sc.learn()
        #bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), algorithm="SAMME", n_estimators=strength)
        #bdt.fit(X, y, sample_weight=1+40*y)
        self._debug("Classifier created")
        self.classifier.label = "Boosted Decision Tree"
        
        
        self.sc2 = SmartClassifier("2",catalog,y,extendeds,remove=['NUMBER','X_IMAGE','Y_IMAGE','WEIGHT','HLR','MAG'])
        self.classifier2 = self.sc2.learn()
                
        #bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), algorithm="SAMME", n_estimators=strength)
        #bdt.fit(X, y, sample_weight=1+40*y)
        self._debug("Classifier2 created")
        self.classifier2.label = "Boosted Decision Tree"
        
        #raise Exception('ha')
        
    
    
    def classify(self, catalog, ellipticity=0.25):
        z = self.sc.classify(catalog) == 1
        gcs = z & (catalog['ELLIPTICITY'] < ellipticity)
        return gcs
        
    def classify2(self, catalog, ellipticity=0.25):
        z = self.sc2.classify(catalog) == 1
        gcs = z & (catalog['ELLIPTICITY'] < ellipticity)
        return gcs
    
        