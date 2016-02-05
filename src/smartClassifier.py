import os
import sys
import numpy as np
import matplotlib, matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from mpl_toolkits.axes_grid1 import make_axes_locatable


class SmartClassifier(object):
    """ Actually still extremely dumb """
    def __init__(self, name, catalog, y, other, remove=[],train=0.4, test=0.35, validate=0.25, debug=True, debugPlot=True):
        self.catalog = catalog.copy()
        self.name = name
        remove.append(y)
        self.remove = remove
        np.random.shuffle(self.catalog)
        self.columns = [a for a in catalog.dtype.names if a not in remove]
        x = self.catalog[self.columns].view(np.float64).reshape(self.catalog.shape + (-1,))
        self.y = y
        
        
        #print(x.shape)
        #yy = y[np.newaxis].T
        #print(yy.shape)
        #xCopy = np.hstack((x,yy))
        #np.random.shuffle(xCopy)
        #self.x = xCopy[:,:-1]
        #self.y = xCopy[:,-1].flatten()
        self.other = other
        self.train= train
        self.test = test
        self.validate = validate
        self.debug = debug
        self.debugPlot = debugPlot
        
        
    def _debug(self, msg):
        if self.debug:
            print(msg)
            
    def classify(self, catalog):
        x = catalog[self.columns].view(np.float64).reshape(catalog.shape + (-1,))
        return self.classifier.predict(x)
        
        
            
        
        
    def learn(self):
        self.classifier = self.getBestClassifier()
        return self.classifier
    
    def evaluate(self, classifier, catalog, highPrecision=False):
        x,y,w = self.getXYW(catalog)
        z = classifier.predict(x)
        tn = ((z == 0) & (y == 0)).sum()
        tp = ((z == 1) & (y == 1)).sum()
        fp = ((z == 1) & (y == 0)).sum()
        fn = ((z == 0) & (y == 1)).sum()
        
        if highPrecision:
            val = tp / (tp + fn + 0.01) + 2.5 * (tn / (fp + tn + 0.01))
            return val
        else:
        
            numerator = (tp * tn) - (fp * fn)
            demoninator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
            if demoninator == 0.0:
                demoninator = 1.0
            mcc = numerator / demoninator
            
            return mcc
    
    def getXYW(self, catalog):
        x = catalog[self.columns].view(np.float64).reshape(catalog.shape + (-1,))
        y = catalog[self.y]
        w = catalog["WEIGHT"]
        return x,y,w
        
    def getBestClassifier(self, highPrecision=False):
        boosts = [1,2,5,10]#,15,20]#,30]#,40]#60,100]#,40,50,70]
        weights = [1,2,3,4]#,6,8]#,8,10,12,16]#,64]#,128]#,12,16,24,32]
        maxDepth = 3
        classifiers = []
        evals = []
        labels = []
        
        numX = self.catalog.shape[0]
        numTrain = np.ceil(numX * self.train)
        numTest = np.ceil(numX * self.test)
        numValidate = np.ceil(numX * self.validate)
        
        train = self.catalog[:numTrain]
        
        test = self.catalog[numTrain:numTrain + numTest]

        validate = self.catalog[numTrain+numTest:]
        
        
        for boost in boosts:
            for i in range(1,maxDepth+1):
                for w in weights:
                    x,y,weight = self.getXYW(train)
                    sw = np.min(np.vstack((train['HLR']/6.0, np.ones(train.shape))), axis=0)**2
                    weight = 1.0*(1-y)+(w-1)*y*sw + 100.0*weight
                    classifier = AdaBoostClassifier(DecisionTreeClassifier(max_depth=i), algorithm="SAMME", n_estimators=boost)
                    classifiers.append(classifier)
                    classifier.fit(x,y, sample_weight=weight)
                    evaluated = self.evaluate(classifier, validate, highPrecision=highPrecision)
                    evals.append(evaluated)
                    label = "BDT,  D=%d,  B=%2d,  W=%2d,  MCC=%0.4f" % (i,boost,w, evaluated)
                    print(label)
                    sys.stdout.flush()
                    labels.append(label)
                    
        evals = np.array(evals)
        print(evals)
        bestClassifierIndex = evals.argmax()
        bestClassifier = classifiers[bestClassifierIndex]
        bestLabel = labels[bestClassifierIndex]
        print(bestLabel)
        self._testClassifier(bestClassifier, test, bestLabel, self.other)
        return bestClassifier
        
    def _testClassifier(self, classifier, catalog, label, other, col0='FALLOFF', col1='CLASS_STAR'):
        self._debug("Testing classifier on validation data for consistency")
        x,y,w = self.getXYW(catalog)
        score = self.evaluate(classifier, catalog)
        z = classifier.predict(x)
        tn = ((z == 0) & (y == 0))
        tp = ((z == 1) & (y == 1))
        fp = ((z == 1) & (y == 0))
        fn = ((z == 0) & (y == 1))
        
        extendedMags = other[:,2]
        extendedHLRs = other[:,3]
        magOffset = 0.52434749611761644
        minMag = max(extendedMags.min(), 11)+magOffset
        maxMag = min(extendedMags.max(), 17)+magOffset
        minHLR = max(extendedHLRs.min(),1)
        maxHLR = min(extendedHLRs.max(),10)
        print(minMag, maxMag, minHLR, maxHLR)
        res = 8
        hlrs = np.linspace(minHLR, maxHLR, res)
        mags = np.linspace(minMag, maxMag, res)
        hTotal, xedg, yedg = np.histogram2d(extendedHLRs, extendedMags+magOffset, bins=[hlrs, mags])
        
        foundHLRs = []
        foundMags = []
        

        for i,row in enumerate(catalog):
            if z[i] == 1 and row[self.y] and row['WEIGHT'] == 0:
                foundHLRs.append(row['HLR'])
                foundMags.append(row['MAG']+magOffset)
        foundHLRs = np.array(foundHLRs)
        foundMags = np.array(foundMags)
        hFound, xedg, yedg = np.histogram2d(foundHLRs, foundMags, bins=[hlrs, mags])
        hRatio = hFound / (hTotal * self.test)
        self.hRatio = hRatio
        self.hlrs = 0.5*(xedg[1:]+xedg[:-1])
        self.mags = 0.5*(yedg[1:]+yedg[:-1])
        if self.debugPlot:
            fig = plt.figure(figsize=(6,6))
            ax0 = fig.add_subplot(1,1,1)
            ax0.invert_yaxis()
            plt.tight_layout()
            h1 = ax0.imshow(hRatio.T, vmax=1, vmin=0,cmap='viridis', interpolation='nearest', extent=[minHLR, maxHLR, maxMag, minMag],aspect=(maxHLR-minHLR)/(maxMag-minMag))

            #ax0.plot(extendedHLRs, extendedMags, 'bo')
            #ax0.plot(foundHLRs, foundMags, 'y.')
            ax0.set_xlabel(r"$r_h\ \rm{(pc)}$", fontsize=18)
            ax0.set_ylabel(r"$\rm{Apparent}\ \rm{magnitude}$", fontsize=18)
            #divider = make_axes_locatable(ax0)
            #cax = divider.append_axes("right", size="5%", pad=0.05)
            #cb = plt.colorbar(h1, cax = cax)
            cb = plt.colorbar(h1,fraction=0.046, pad=0.03)
            cb.set_label(r"$\rm{Probability}$", fontsize=18)
            fig.savefig("classifier_%s.pdf"%self.name, bbox_inches="tight")
            np.save("classifier_%s.npy"%self.name, hRatio)
            plt.show()
            
        
        
        self._debug("Matthews correlation corefficient: %0.4f" % (score))
        self._debug("Extended correct: %0.1f%%" % (100.0 * tp.sum() / (y==1).sum()))
        self._debug("False negative: %0.1f%%" % (100.0 * fn.sum() / (y==1).sum()))
        self._debug("Point correct: %0.1f%%" % (100.0 * tn.sum() / (y==0).sum()))
        self._debug("False positive: %0.1f%%" % (100.0 * fp.sum() / (y==0).sum()))
        self._debug("Num True positive / Num Positive: %0.3f" % (1.0 * tp.sum() / (tp.sum() + fp.sum())))
        self._debug("Num True positive / Num Actual Positive: %0.3f" % (1.0 * tp.sum() / y.sum()))
        
        if self.debugPlot:
            fig = plt.figure(figsize=(8,8))
            ax0 = fig.add_subplot(1,1,1)
            ax0.scatter(catalog[tn][col0], catalog[tn][col1], label="Correct Point", c="k", lw=0, alpha=0.1)
            ax0.scatter(catalog[tp][col0], catalog[tp][col1], label="Correct Extended", c="g", lw=0, s=30, alpha=0.4)
            ax0.scatter(catalog[fn][col0], catalog[fn][col1], label="False negative", c="r", marker="+", lw=1, s=30, alpha=0.4)
            ax0.scatter(catalog[fp][col0], catalog[fp][col1], label="False positive", c="m", lw=1, marker="x", s=20, alpha=0.4)
            ax0.legend()
            ax0.set_xlabel("FALLOFF")
            ax0.set_ylabel("CLASS_STAR")
            ax0.set_title(label)
            plt.show()
            
