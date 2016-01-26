import os
import sys
import numpy as np
import matplotlib, matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier


class SmartClassifier(object):
    """ Actually still extremely dumb """
    def __init__(self, x, y, train=0.5, test=0.25, validate=0.25, debug=True, debugPlot=True):
        self.x = x
        self.y = y
        self.train= train
        self.test = test
        self.validate = validate
        self.debug = debug
        self.debugPlot = debugPlot
        
        
    def _debug(self, msg):
        if self.debug:
            print(msg)
            
    def learn(self):
        classifier = self.getBestClassifier()
        return classifier
    
    def evaluate(self, classifier, validateX, y, highPrecision=False):
        z = classifier.predict(validateX)
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
    
    def getBestClassifier(self, highPrecision=False):
        boosts = [1,2,5,10,15,20]#,30]#,40]#60,100]#,40,50,70]
        weights = [4,8,10,12,16]#,64]#,128]#,12,16,24,32]
        classifiers = []
        evals = []
        labels = []
        
        numX = self.x.shape[0]
        numTrain = np.ceil(numX * self.train)
        numTest = np.ceil(numX * self.test)
        numValidate = np.ceil(numX * self.validate)
        
        trainX = self.x[:numTrain,:]
        trainY = self.y[:numTrain]
        
        testX = self.x[numTrain:numTrain + numTest, :]
        testY = self.y[numTrain:numTrain + numTest]

        validateX = self.x[numTrain+numTest:,:]
        validateY = self.y[numTrain + numTest:]  
        
        
        for boost in boosts:
            for i in range(1,4):
                for w in weights:
                    weight = 1+(w-1)*trainY
                    classifier = AdaBoostClassifier(DecisionTreeClassifier(max_depth=i), algorithm="SAMME", n_estimators=boost)
                    classifiers.append(classifier)
                    classifier.fit(trainX, trainY, sample_weight=weight)
                    evaluated = self.evaluate(classifier, validateX, validateY, highPrecision=highPrecision)
                    evals.append(evaluated)
                    label = "Boosted tree, depth %d with %d boosts using weight %d with Matthews Correlation coefficient %0.4f" % (i,boost,w, evaluated)
                    print(label)
                    sys.stdout.flush()
                    labels.append(label)
                    
        evals = np.array(evals)
        print(evals)
        bestClassifierIndex = evals.argmax()
        bestClassifier = classifiers[bestClassifierIndex]
        bestLabel = labels[bestClassifierIndex]
        print(bestLabel)
        self._testClassifier(bestClassifier, testX, testY, bestLabel)
        return bestClassifier
        
    def _testClassifier(self, classifier, X, y, label, col0=7, col1=6):


        self._debug("Testing classifier on validation data for consistency")
        score = self.evaluate(classifier, X, y)
        z = classifier.predict(X)
        tn = ((z == 0) & (y == 0))
        tp = ((z == 1) & (y == 1))
        fp = ((z == 1) & (y == 0))
        fn = ((z == 0) & (y == 1))
        
        
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
            ax0.scatter(X[tn, col0], X[tn, col1], label="Correct Point", c="k", lw=0, alpha=0.1)
            ax0.scatter(X[tp, col0], X[tp, col1], label="Correct Extended", c="g", lw=0, s=30)
            ax0.scatter(X[fn, col0], X[fn, col1], label="False negative", c="r", marker="+", lw=1, s=30)
            ax0.scatter(X[fp, col0], X[fp, col1], label="False positive", c="m", lw=1, marker="x", s=20)
            ax0.legend()
            #ax0.set_xlabel("FALLOFF")
            #ax0.set_ylabel("CLASS_STAR")
            ax0.set_title(label)
            plt.show()