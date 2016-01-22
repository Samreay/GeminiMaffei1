import os
from classifier import *
from reducer import *
from tile import *
from helper import *
from mosaic import *

tempParentDir = "../temp"
tileDir = "../resources/tiles"
mosaicDir = "../resources/mosaic"


classifier = Classifier("../resources/classified/candidates.fits", "../resources/classified/candidates.txt", tempParentDir=tempParentDir, debugPlot=True)
classifier.getClassifier()
'''

tiles = [Tile(os.path.abspath(tileDir + os.sep + f), classifier, tempParentDir=tempParentDir) for f in os.listdir(tileDir) if os.path.isfile(tileDir + os.sep + f) and f.endswith(".fits")]

gcs = None
for tile in tiles:
    print("Getting GCs for tile %s" % tile.subName)
    newGCs = tile.getGlobalClusters()
    if gcs is None:
        gcs = newGCs
    else:
        gcs = np.concatenate((gcs, newGCs))



mosaics = [Mosaic(os.path.abspath(mosaicDir + os.sep + f), tempParentDir=tempParentDir) for f in os.listdir(mosaicDir) if f.endswith(".fits")]
mosaics = [mosaics[2]]
for mosaic in mosaics:
    mosaic.importCatalog(gcs)
    mosaic.show()
'''