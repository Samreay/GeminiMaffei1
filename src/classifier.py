import os
import sys
import numpy as np
import matplotlib, matplotlib.pyplot as plt
import sextractor
from astropy.io import fits
from scipy.ndimage.filters import *
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from reducer import Reducer

class Classifier(Reducer):
    def __init__(self, fitsFile, candidates, debug=True, debugPlot=False, tempParentDir=None):
        self.fitsFile = fitsFile
        self.candidates = candidates
        self.classifier = None
        super(Classifier, self).__init__(debug=debug, debugPlot=debugPlot, tempParentDir=tempParentDir, tempSubDir="classifier")
       
        
    def getClassifier(self):
        self._setupTempDir()
        catalog = self.getCatalog(self.fitsFile)
        self.candidateMask = self._getCandidateMask(catalog, self.candidates)
        self.classifier = self._trainClassifier(catalog, self.candidateMask)
        self._testClassifier(catalog, self.candidateMask)
        self._cleanTempDir()
        self._debug("Classifier generated. Now you can invoke .clasify(catalog)")

    def _getCandidateMask(self, catalog, candidates, pixelThreshold=2):
        self._debug("Loading in Ricardo's extendeds")
        coord = np.loadtxt(candidates)
        mask = []
        for entry in catalog:
            dists = np.sqrt((entry['X_IMAGE'] - coord[:,0])**2 + (entry['Y_IMAGE'] - coord[:,1])**2 )
            mask.append(dists.min() < pixelThreshold)
        mask = np.array(mask)
        return mask 
        
    def _trainClassifier(self, catalog, candidateMask, strength=15):
        y = candidateMask * 1
        X = catalog.view(np.float64).reshape(catalog.shape + (-1,))[:, 3:]
        
        self._debug("Creating classifier")
        # Create and fit an AdaBoosted decision tree
        bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), algorithm="SAMME", n_estimators=strength)
        bdt.fit(X, y, sample_weight=1+40*y)
        self._debug("Classifier created")
        bdt.label = "Boosted Decision Tree"
        return bdt
    
    
    def classify(self, catalog, ellipticity=0.25):
        X = catalog.view(np.float64).reshape(catalog.shape + (-1,))[:, 3:]
        z = self.classifier.predict(X) == 1
        gcs = z & (catalog['ELLIPTICITY'] < ellipticity)
        return gcs
    
    def _testClassifier(self, catalog, candidateMask):
        X = catalog.view(np.float64).reshape(catalog.shape + (-1,))[:, 3:]
        y = candidateMask

        self._debug("Testing classifier on training data for consistency")
        z = self.classifier.predict(X)
        correctPoint = (z == 0) & (~y)
        correctExtended = (z == 1) & (y)
        incorrectPoint = (z == 1) & (~y)
        incorrectExtended = (z == 0) & (y)
        
        
        self._debug("Extended correct: %0.1f%%" % (100.0 * correctExtended.sum() / y.sum()))
        self._debug("False negative: %0.1f%%" % (100.0 * incorrectExtended.sum() / (y).sum()))
        self._debug("Point correct: %0.1f%%" % (100.0 * correctPoint.sum() / (~y).sum()))
        self._debug("False positive: %0.1f%%" % (100.0 * incorrectPoint.sum() / (~y).sum()))
        
        if self.debugPlot:
            fig = plt.figure(figsize=(8,8))
            ax0 = fig.add_subplot(1,1,1)
            ax0.scatter(catalog[correctPoint]['FALLOFF'], catalog[correctPoint]['CLASS_STAR'], label="Correct Point", c="k", lw=0, alpha=0.1)
            ax0.scatter(catalog[correctExtended]['FALLOFF'], catalog[correctExtended]['CLASS_STAR'], label="Correct Extended", c="g", lw=0, s=30)
            ax0.scatter(catalog[incorrectExtended]['FALLOFF'], catalog[incorrectExtended]['CLASS_STAR'], label="False negative", c="r", marker="+", lw=1, s=30)
            ax0.scatter(catalog[incorrectPoint]['FALLOFF'], catalog[incorrectPoint]['CLASS_STAR'], label="False positive", c="m", lw=1, marker="x", s=20)
            ax0.legend()
            ax0.set_xlabel("FALLOFF")
            ax0.set_ylabel("CLASS_STAR")
            ax0.set_title(self.classifier.label)
            plt.show()