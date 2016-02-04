import os
from classifier import *
from reducer import *
from tile import *
from helper import *
from mosaic import *
from dust import Dust
from numpy.lib.recfunctions import append_fields

tempParentDir = "../temp"
tileDir = "../resources/tiles"
mosaicDir = "../resources/mosaic"
view = r"/Users/shinton/Documents/backup/GeminiMaffei1/resources/mosaicZSub.fits"

'''
classifier = Classifier("../resources/classified/candidates.fits", "../resources/classified/candidates.txt", tempParentDir=tempParentDir, debugPlot=True)
classifier.getClassifiers()


tiles = [Tile(os.path.abspath(tileDir + os.sep + f), classifier, tempParentDir=tempParentDir) for f in os.listdir(tileDir) if os.path.isfile(tileDir + os.sep + f) and f.endswith(".fits")]

gcs = None
psfs = None
for tile in tiles:
    print("Getting GCs for tile %s" % tile.subName)
    newGCs = tile.getGlobalClusters()
    cat = tile.getCatalog(tile.fitsPath)
    psfList = tile.getPSFStars(tile.fitsPath, tile.catalog)
    psf = cat[psfList[0] | psfList[1]]
    psf = tile.getRAandDec(tile.fitsPath, psf)
    
    if gcs is None:
        gcs = newGCs
    else:
        gcs = np.concatenate((gcs, newGCs))
    if psfs is None:
        psfs = psf
    else:
        print(psfs.shape)
        print(psf.shape)
        
        psfs = np.concatenate((psfs, psf))



mosaics = [Mosaic(os.path.abspath(mosaicDir + os.sep + f), tempParentDir=tempParentDir) for f in os.listdir(mosaicDir) if f.endswith(".fits")]
mosaicZ = [i for i in mosaics if i.subName.find("_Z") > 0]

# This grabs mag best using the z band only (or more than that if I want)
apGCs = gcs.copy()
for mosaic in mosaicZ:
    print(mosaic.subName)
    mosaic.importCatalog(apGCs)
    mags, magEs = mosaic.getMagnitudes()
    prefix = mosaic.subName[mosaic.subName.index("_")+1:]
    apGCs = append_fields(apGCs, prefix+"_MAG", mags, usemask=False)    
    apGCs = append_fields(apGCs, prefix+"_MAGE", magEs, usemask=False)    
    apGCs = append_fields(apGCs, prefix+"_MASK", ((np.isfinite(apGCs[prefix+'_MAG'])) & (apGCs[prefix+'_MAG'] < 99) & (apGCs[prefix+'_MAGE'] < 20)), dtypes=[np.bool], usemask=False)

print("Done mosaic z")

# From the ra and dec given by the z band, use phot to do photometry
gc = apGCs.copy()
for mosaic in mosaics:
    gc = mosaic.getPhot(gc)

psfsColour = psfs.copy()
for mosaic in mosaics:
    psfsColour = mosaic.getPhot(psfsColour)


dust = Dust()
gcc = dust.correctExtinctions(gc.copy())
psfc = dust.correctExtinctions(psfsColour.copy())
'''
gccd = addColourDiff(gcc)
psfd = addColourDiff(psfc)

gccdd = removeDoubles(gccd)

print(gccdd.shape)
gccf = plotColourDifference(gccdd, psfd)
print("a",gccf.shape)
gccf = gccf[gccf['KingFWHM'] < 15]
print("b",gccf.shape)
gccf = gccf[gccf['Z_ABS'] > -10.5]
print("c",gccf.shape)
gccf = gccf[np.abs(gccf['RMZ_9']) < 4]
print("d",gccf.shape)
gccf = gccf[gccf['KingFWHM'] > 0.1]
print("e",gccf.shape)

gccf = addFWHM(gccf)

colors = ['Chi2DeltaKingDiv']#, 'ELLIPTICITY', 'CI', 'CI2', 'KingFWHM']
for c in colors:
    plotColourDiagrams(gccf, colourColumn=c)    

plotSizeDiagrams(gccf)    


classA = gccf['Chi2DeltaKingDiv'] > 1.5
classB = ~classA

print(classA.sum())
print(classB.sum())



#print(latexPrint(gccf[classA][:10], "A"))
#print("\n\n---\n\n")
#print(latexPrint(gccf[classB][:10], "B"))

