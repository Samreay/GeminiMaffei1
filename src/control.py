import os
from classifier import *
from reducer import *
from tile import *
from helper import *
from mosaic import *
from numpy.lib.recfunctions import append_fields

tempParentDir = "../temp"
tileDir = "../resources/tiles"
mosaicDir = "../resources/mosaic"

'''
classifier = Classifier("../resources/classified/candidates.fits", "../resources/classified/candidates.txt", tempParentDir=tempParentDir, debugPlot=True)
classifier.getClassifiers()


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
#mosaics = [mosaics[2]]
apGCs = gcs.copy()
for mosaic in mosaics:
    mosaic.importCatalog(apGCs)
    mags, magEs = mosaic.getMagnitudes()
    prefix = mosaic.subName[mosaic.subName.index("_")+1:]
    apGCs = append_fields(apGCs, prefix+"_MAG", mags, usemask=False)    
    apGCs = append_fields(apGCs, prefix+"_MAGE", magEs, usemask=False)    
    apGCs = append_fields(apGCs, prefix+"_MASK", ((np.isfinite(apGCs[prefix+'_MAG'])) & (apGCs[prefix+'_MAG'] < 99) & (apGCs[prefix+'_MAGE'] < 20)), dtypes=[np.bool], usemask=False)

'''
    
# ADD COLOUR DIAGRAMS
plotColourDiagrams(apGCs)    
    
#mosaics[2].show()
