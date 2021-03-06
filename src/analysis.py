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
debugFlag = True
debugPlot = True
urekaPath = "/Users/shinton/Software/Ureka"
wcsToolsPath = "/Users/shinton/Software/wcstools/bin"
baolabScript = '/Users/shinton/Software/baolab-0.94.1e/als2bl.cl'
sexPath = '/Users/shinton/Software/Ureka/bin/sex'
scampPath = '/usr/local/bin/scamp'
missfitsPath = '/usr/local/bin/missfits'
fitsPath = "../resources/clarisse.fits"
mosaicFileI = "../resources/all_I.fits"
mosaicFileR = "../resources/all_R.fits"
mosaicFileZ = "../resources/all_Z.fits"
cmap = "Blues_r"

def debug(msg):
    if debugFlag:
        print(msg)
        sys.stdout.flush()


def getFilters():
    dirName = "../resources/filters"
    filters = {}
    for f in os.listdir(dirName):
        if f.endswith(".conv"):
            filters[f[:-5]] = np.loadtxt(dirName + os.sep + f, skiprows=1)
    return filters
    
def getDefaultSextractor():
    # Create a SExtractor instance
    sex = sextractor.SExtractor()
    
    # Modify the SExtractor configuration
    sex.config['PIXEL_SCALE'] = 0.2
    sex.config['VERBOSE_TYPE'] = "QUIET"
    sex.config['MAG_ZEROPOINT'] = 25.0
    sex.config['BACK_SIZE'] = 256
    sex.config['BACK_TYPE'] = "MANUAL"
    #sex.config['BACK_VALUE'] = 0
    sex.config['BACKPHOTO_TYPE'] = "LOCAL"
    sex.config['BACK_FILTERSIZE'] = 3
    sex.config['DETECT_THRESH'] = 2
    sex.config['ANALYSIS_THRESH'] = 2
    sex.config['SEEING_FWHM'] = 0.6
    sex.config['GAIN'] = 5.0
    sex.config['SATUR_LEVEL'] = 45000
    sex.config['MAG_GAMMA'] = 4.0
    sex.config['MEMORY_OBJSTACK'] = 30000
    sex.config['MEMORY_PIXSTACK'] = 6000000
    sex.config['MEMORY_BUFSIZE'] = 16384
    sex.config['DEBLEND_MINCONT'] = 0.001
    return sex

def getScampSextractor():
    sex = getDefaultSextractor()
    sex.config['CATALOG_TYPE'] = 'FITS_LDAC'
    sex.config['PARAMETERS_LIST'] = ['NUMBER','X_IMAGE','Y_IMAGE','XWIN_IMAGE','YWIN_IMAGE','ELLIPTICITY','ERRAWIN_IMAGE','ERRBWIN_IMAGE','ERRTHETAWIN_IMAGE','FLUX_AUTO','FLUXERR_AUTO','FLAGS','FLAGS_WEIGHT','FLUX_RADIUS']
    filters = getFilters()
    sex.config['FILTER_MASK'] = filters['gauss_3.0_5x5']

    return sex

def getSextractor(**kwargs):
    sex = getDefaultSextractor()
    sex.config['PHOT_APERTURES'] = [1,2,2.5,3,3.25,3.5,3.75,4,4.25,4.5,5,5.5,6,6.5,7,8,12,14,18,20,25]
    sex.config['CHECKIMAGE_TYPE'] = ["BACKGROUND", "APERTURES"]
    sex.config['CHECKIMAGE_NAME'] = ["back.fits", "apertures.fits"]
    sex.config['PARAMETERS_LIST'] = ['NUMBER', 'X_IMAGE', 'Y_IMAGE'] + ['MAG_APER(%d)'%(i+1) for i in range(len(sex.config['PHOT_APERTURES']))] + ['FLUX_MAX', 'FLUX_AUTO', 'MAG_AUTO', 'ELLIPTICITY', 'FWHM_IMAGE', 'MU_MAX', 'CLASS_STAR']
    for (key, value) in kwargs.iteritems():
        sex.config[key] = value
    return sex

def getInitialObjects():
    filters = getFilters()
    imageName = "../resources/clarisse+sky.fits"
    sex = getSextractor()
    sex.config['FILTER_MASK'] = filters['gauss_3.0_5x5']
    data = sex.run(imageName, path=sexPath)
    return data

def getDefaultImgComparison():
    fig = plt.figure(figsize=(14,10))
    ax0 = fig.add_subplot(1,2,1)
    ax1 = fig.add_subplot(1,2,2)
    ax0.set_title("Orginal data")
    ax1.set_title("Median subtracted")
    return fig,ax0,ax1

def addColourBars(inps):
    for handle, axis in inps:
        divider0 = make_axes_locatable(axis)
        cax0 = divider0.append_axes("right", size="4%", pad=0.05)
        plt.colorbar(handle, cax=cax0)
    
def getMaskedImage(imageStart):
    debug("Masking image")
    median = np.median(imageStart)
    badPixels = (imageStart < -1000) | (imageStart > 1e9)
    imageFixed = imageStart * (1 - badPixels) + median * badPixels
    
    if debugPlot:
        fig, ax0, ax1 = getDefaultImgComparison()
        ax0.set_title("Input image")
        ax1.set_title("Masked image")
        im0 = ax0.imshow(imageStart, vmin=1300, vmax=1400, cmap=cmap, origin="lower")
        im1 = ax1.imshow(imageFixed, vmin=1300, vmax=1400, cmap=cmap, origin="lower")
        addColourBars([(im0, ax0), (im1, ax1)])
        plt.show()
    debug("Image masked")
    return badPixels, imageFixed
    
    
    
def broadClean(originalImage, redo=False, fast=False, s=150):
    outputName = "../resources/clarisseBroadClean.fits"
    analyse = True
    if not redo and os.path.isfile(outputName):
        try:
            f = fits.open(outputName)
            filterImage = f[0].data
            f.close()
            debug("Loaded existing file at %s" % outputName)
            analyse = False
        except Exception:
            debug("Previous file at %s not found" % outputName)
    if analyse:
        debug("Generating broad clean fits file")
        if fast:
            ratio = 0.5
            debug("Fast option selected. Reducing resolution by factor of %0.2f" % (1/ratio))
            s = np.ceil(s * ratio)
            image = resize(originalImage, np.floor(np.array(originalImage.shape) * ratio), order=1, preserve_range=True)
            debug("Image resized")
        else:
            image = originalImage
        y = np.arange(-s+1,s)[:,np.newaxis]
        x = np.arange(-s+1,s)[np.newaxis,:]
        r = np.sqrt(x*x + y*y)
        fil = ((r < s) & (r > np.floor(s * 0.7)))
        debug("Beginning convolution")
        t = time()
        filterImage = median_filter(image, footprint=fil)
        debug("Convolution completed in %0.2f seconds" % (time() - t))
        if fast:
            debug("Expanding image to original size")
            filterImage = resize(filterImage, originalImage.shape, order=1, preserve_range=True) #scipy.ndimage.interpolation.zoom
        
    if debugPlot:
        fig, ax0, ax1 = getDefaultImgComparison()
        ax0.set_title("Input image")
        ax1.set_title("Median filter image")
        im0 = ax0.imshow(originalImage, vmin=1310, vmax=1450, cmap=cmap, origin="lower")
        im1 = ax1.imshow(filterImage, vmin=1310, vmax=1450, cmap=cmap, origin="lower")
        addColourBars([(im0, ax0), (im1, ax1)])
        plt.show()
    if analyse:
        hdu = fits.PrimaryHDU(filterImage)
        if os.path.isfile(outputName):
            debug("Existing broad clean file found. Overwriting.")
        hdu.writeto(outputName, clobber=True)
        debug("Broad clean fits file generated and saved to %s" % outputName)
    return (outputName, filterImage)
    
    
def getSubtracted(originalImage, cleanImage, fitsPath):
    result = originalImage - cleanImage
    result[result > 1e10] = 0
    debug("Subtracting clean background")
    if debugPlot:
        fig, ax0, ax1 = getDefaultImgComparison()
        ax0.set_title("Input image")
        ax1.set_title("Background subtracted image")
        im0 = ax0.imshow(originalImage, vmin=1310, vmax=1450, cmap=cmap, origin="lower")
        im1 = ax1.imshow(result, vmin=0, vmax=100, cmap=cmap, origin="lower")
        addColourBars([(im0, ax0), (im1, ax1)])
        plt.show()
    debug("Background subtracted")
    saveas = "../resources/subtracted.fits"
    fitsFile = fits.open(fitsPath)
    fitsFile[0].data = result
    fitsFile.writeto(saveas, clobber=True)
    fitsFile.close()
    debug("Subtracted fits file saved to %s for testing" % saveas)
    return result
    
    
    
def addSkyFlux(imageOriginal, imageClean, useMax=True):
    mask = (imageOriginal < 1e6)
    numBins = imageClean.size / 1000 # magic number, 1000 pixels per bin on uniform
    hist, edges = np.histogram(imageOriginal[mask], bins=numBins, density=True)
    centers = 0.5 * (edges[1:] + edges[:-1])
    debug("Filtering original image an constructing flux probability using %d bins" % numBins)
    result = centers[hist.argmax()]
    result2 = np.median(imageOriginal)
    debug("Max likelihood flux: %0.2f. Median flux: %0.2f" % (result, result2))
    imageOutput = imageClean + result
    
    
    if debugPlot:
        fig = plt.figure(figsize=(10,4))
        ax0 = fig.add_subplot(1,1,1)
        conv = [np.exp(-(x*x*0.1)) for x in np.arange(-5,6)]
        data = np.convolve(hist, conv, mode='same')
        ax0.plot(centers, np.log(data))
        ax0.set_xlabel("flux")
        ax0.set_ylabel(r"$\log\left(P(\rm{flux})\right)$", fontsize=16)
        ax0.axvline(result, color='r', linestyle='--', label="Sky", alpha=0.7)
        ax0.legend()
        plt.show()

    debug("Sky flux found to be %0.2f" % result)
    return (result, imageOutput)
        
    
    
    
def getCatalogs(fitsPath, imageClean, pixelThreshold=2):
    # Load in the original fits file which has all the right header data, modify the image and save to temp
    debug("Getting objects using sextractor")
    try:
        tempDir = tempfile.mkdtemp()
        tempFits = tempDir + os.sep + "temp.fits"
        debug("Generating temp directory at %s" % tempDir)
        fitsFile = fits.open(fitsPath)
        fitsFile[0].data = imageClean
        fitsFile.writeto(tempFits)
        fitsFile.close()
        
        filters = getFilters()
        sex = getSextractor()
        sex.config['FILTER_MASK'] = filters['gauss_4.0_7x7']
        debug("Running sextractor using gaussian filter")
        data1 = sex.run(tempFits, path=sexPath, cwd=tempDir)
        sex.config['FILTER_MASK'] = filters['mexhat_2.5_7x7']
        debug("Running sextractor using mexhat filter")
        data2 = sex.run(tempFits, path=sexPath, cwd=tempDir)
        
        cat1 = data1['catalog']
        cat2 = data2['catalog']
        
        debug("Merging catalogs")
        mask = []
        for entry in cat2:
            x = entry['X_IMAGE']
            y = entry['Y_IMAGE']
            xdiff = cat1['X_IMAGE'] - x
            xdiff *= xdiff
            ydiff = cat1['Y_IMAGE'] - y
            ydiff *= ydiff
            rdiff = xdiff + ydiff
            rmin= rdiff.min()
            mask.append(rmin > pixelThreshold)
        mask = np.array(mask)
        catTotal = np.concatenate((cat1, cat2[mask]))
        debug("Catalogs merged. Found %d objects." % catTotal.size)
    
        return catTotal, sex
             
    except:
        raise
    finally:
        try:
            shutil.rmtree(tempDir)  # delete directory
            pass
        except OSError as exc:
            if exc.errno != errno.ENOENT:
                raise  # re-raise exception

def visualiseHigherDimensions(catalog, maskExtended):
    
    debug("Checking for data patterns. Reducing dimensionality for visualisation")
    cat = catalog.view(np.float64).reshape(catalog.shape + (-1,))
    sphere_data = cat[:, 3:]
    # Variables for manifold learning.
    s = 1000
    downFilterIndexes = np.unique(np.concatenate((np.random.randint(sphere_data.shape[0], size=s), np.where(maskExtended)[0])))
    maskExtended = maskExtended[downFilterIndexes]
    sphere_data = sphere_data[downFilterIndexes, :]
    n_neighbors = 10
    colors = maskExtended * 100

    fig = plt.figure(figsize=(20, 12))
    # Perform Locally Linear Embedding Manifold learning
    methods = ['standard', 'ltsa', 'hessian', 'modified']
    labels = ['LLE', 'LTSA', 'Hessian LLE', 'Modified LLE']
    
    


    for i, method in enumerate(methods):
        try:
            ax = fig.add_subplot(252 + i)
            plt.title("%s" % (labels[i]))
            t0 = time()
            trans_data = manifold.LocallyLinearEmbedding(n_neighbors, 2,
                                    method=method).fit_transform(sphere_data).T
            t1 = time()
            debug("%s: %.2g sec" % (methods[i], t1 - t0))
            plt.scatter(trans_data[0][maskExtended], trans_data[1][maskExtended], c='r', cmap=plt.cm.rainbow,marker='>')
            plt.scatter(trans_data[0][~maskExtended], trans_data[1][~maskExtended], c='b', cmap=plt.cm.rainbow,marker='o', alpha=0.2)
            plt.title("%s (%.2g sec)" % (labels[i], t1 - t0))
            ax.xaxis.set_major_formatter(NullFormatter())
            ax.yaxis.set_major_formatter(NullFormatter())
            plt.axis('tight')
        except:
            pass
    
    # Perform Isomap Manifold learning.
    t0 = time()
    trans_data = manifold.Isomap(n_neighbors, n_components=2)\
        .fit_transform(sphere_data).T
    t1 = time()
    debug("%s: %.2g sec" % ('ISO', t1 - t0))
    
    ax = fig.add_subplot(257)
    plt.scatter(trans_data[0][maskExtended], trans_data[1][maskExtended], c='r', cmap=plt.cm.rainbow, marker='>')
    plt.scatter(trans_data[0][~maskExtended], trans_data[1][~maskExtended], c='b', cmap=plt.cm.rainbow, marker='o', alpha=0.2)
    plt.title("%s (%.2g sec)" % ('Isomap', t1 - t0))
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    plt.axis('tight')
    
    # Perform Multi-dimensional scaling.
    t0 = time()
    mds = manifold.MDS(2, max_iter=100, n_init=1)
    trans_data = mds.fit_transform(sphere_data).T
    t1 = time()
    debug("MDS: %.2g sec" % (t1 - t0))
    
    ax = fig.add_subplot(258)
    plt.scatter(trans_data[0][maskExtended], trans_data[1][maskExtended], c='r', cmap=plt.cm.rainbow, marker='>')
    plt.scatter(trans_data[0][~maskExtended], trans_data[1][~maskExtended], c='b', cmap=plt.cm.rainbow, marker='o', alpha=0.2)
    plt.title("MDS (%.2g sec)" % (t1 - t0))
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    plt.axis('tight')
    
    # Perform Spectral Embedding.
    t0 = time()
    se = manifold.SpectralEmbedding(n_components=2,
                                    n_neighbors=n_neighbors)
    trans_data = se.fit_transform(sphere_data).T
    t1 = time()
    debug("Spectral Embedding: %.2g sec" % (t1 - t0))
    
    ax = fig.add_subplot(259)
    plt.scatter(trans_data[0][maskExtended], trans_data[1][maskExtended], c='r', cmap=plt.cm.rainbow, marker='>')
    plt.scatter(trans_data[0][~maskExtended], trans_data[1][~maskExtended], c='b', cmap=plt.cm.rainbow, marker='o', alpha=0.2)
    plt.title("Spectral Embedding (%.2g sec)" % (t1 - t0))
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    plt.axis('tight')
    
    # Perform t-distributed stochastic neighbor embedding.
    t0 = time()
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    trans_data = tsne.fit_transform(sphere_data).T
    t1 = time()
    debug("t-SNE: %.2g sec" % (t1 - t0))
    
    ax = fig.add_subplot(2, 5, 10)
    plt.scatter(trans_data[0][maskExtended], trans_data[1][maskExtended], c='r', cmap=plt.cm.rainbow, marker=">")
    plt.scatter(trans_data[0][~maskExtended], trans_data[1][~maskExtended], c='b', cmap=plt.cm.rainbow, marker="o", alpha=0.2)
    plt.title("t-SNE (%.2g sec)" % (t1 - t0))
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    plt.axis('tight')
    plt.show()

def checkForClustering(catalog):
    debug("Checking for data clustering")
    Xfull = catalog.view(np.float64).reshape(catalog.shape + (-1,))[:,1:]
    X = Xfull[:,2:]
    
    
    debug("Using DBSCAN")
    db = DBSCAN(eps=0.3, min_samples=10).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    n_clusters_DBSCAN = len(set(labels)) - (1 if -1 in labels else 0)
    debug('Estimated number of clusters with DBSCAN: %d' % n_clusters_DBSCAN)
        
    unique_labelsDBSCAN = set(labels)
    colorsDBSCAN = plt.cm.rainbow(np.linspace(0, 1, len(unique_labelsDBSCAN)))
    
    debug("Estimating clusters using MeanShift")
    bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=500)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(X)
    labelsMS = ms.labels_
    cluster_centers = ms.cluster_centers_
    labels_uniqueMS = np.unique(labelsMS)
    n_clusters_MS = len(labels_uniqueMS)
    debug("Estimated number of clusters with MeanShift: %d" % n_clusters_MS)
    
    # Plot result
    fig = plt.figure(figsize=(12,12))
    ax0 = fig.add_subplot(2,2,1)
    ax1 = fig.add_subplot(2,2,2)
    ax2 = fig.add_subplot(2,2,3)
    ax3 = fig.add_subplot(2,2,4)
    for k, col in zip(unique_labelsDBSCAN, colorsDBSCAN):
        if k == -1:
            col = 'k'
        class_member_mask = (labels == k)
        mask = class_member_mask & core_samples_mask
        xy = Xfull[mask]
        ax0.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=5)
        ax2.plot(catalog['MAG_APER(1)'][mask], catalog['CLASS_STAR'][mask], 'o', markerfacecolor=col, markeredgecolor='k', markersize=5)
        xy = Xfull[class_member_mask & ~core_samples_mask]
        ax0.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=5)
        ax2.plot(catalog['MAG_APER(1)'][class_member_mask & ~core_samples_mask], catalog['CLASS_STAR'][class_member_mask & ~core_samples_mask], 'o', markerfacecolor=col, markeredgecolor='k', markersize=5)

        ax0.set_title('DBCAN: # clusters: %d' % n_clusters_DBSCAN)
        
        
    colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
    for k, col in zip(range(n_clusters_MS), colors):
        my_members = labelsMS == k
        cluster_center = cluster_centers[k]
        ax1.plot(Xfull[my_members, 0], Xfull[my_members, 1], col + '.')
        ax3.plot(catalog['MAG_APER(1)'][my_members], catalog['CLASS_STAR'][my_members], col + '.')
        #ax1.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=14)
    ax1.set_title('MeanShift: # clusters: %d' % n_clusters_MS)
    plt.show()

def showStats(catalog, m):
    
    cat = catalog.view(np.float64).reshape(catalog.shape + (-1,))
    cat = cat[:, 1:]    
        
    fig = plt.figure(figsize(10,8))
    ax0 = fig.add_subplot(2,2,1)
    ax1 = fig.add_subplot(2,2,2)
    ax2 = fig.add_subplot(2,2,3)
    ax3 = fig.add_subplot(2,2,4)
    #ax4 = fig.add_subplot(2,3,5)
    #ax5 = fig.add_subplot(2,3,6)
    
    ax0.scatter(catalog['X_IMAGE'][m], catalog['Y_IMAGE'][m], lw = 0, c=catalog['CLASS_STAR'][m], marker='>')
    ax0.scatter(catalog['X_IMAGE'][~m], catalog['Y_IMAGE'][~m], lw = 0, c=catalog['CLASS_STAR'][~m], marker='o', alpha=0.2)
    ax0.set_xlabel("X")
    ax0.set_ylabel("Y")
    
    ax1.scatter(catalog['MAG_AUTO'][m], catalog['CLASS_STAR'][m], c=catalog['CLASS_STAR'][m], lw=0, marker='>')
    ax1.scatter(catalog['MAG_AUTO'][~m], catalog['CLASS_STAR'][~m], alpha=0.02, c=catalog['CLASS_STAR'][~m], lw=0, marker='o')
    ax1.set_xlabel("MAG_AUTO")
    ax1.set_ylabel("CLASS_STAR")
    
    ax2.scatter(catalog['MAG_AUTO'][m], catalog['FALLOFF'][m], c=catalog['CLASS_STAR'][m], lw=0, marker='>')
    ax2.scatter(catalog['MAG_AUTO'][~m], catalog['FALLOFF'][~m], alpha=0.02, c=catalog['CLASS_STAR'][~m], lw=0, marker='o')
    ax2.set_xlabel("MAG_AUTO")
    ax2.set_ylabel("FALLOFF")
    
    ax3.scatter(catalog['FALLOFF'][m], catalog['CLASS_STAR'][m], c=catalog['CLASS_STAR'][m], lw=0, marker='>')
    ax3.scatter(catalog['FALLOFF'][~m], catalog['CLASS_STAR'][~m], alpha=0.02, c=catalog['CLASS_STAR'][~m], lw=0, marker='o')
    ax3.set_xlabel("FALLOFF")
    ax3.set_ylabel("CLASS_STAR")

    '''
    ax1.scatter(catalog['MAG_APER(1)'][maskExtended], catalog['CLASS_STAR'][maskExtended], c=catalog['CLASS_STAR'][maskExtended], lw=0, marker='>')
    ax1.scatter(catalog['MAG_APER(1)'][~maskExtended], catalog['CLASS_STAR'][~maskExtended], alpha=0.02, c=catalog['CLASS_STAR'][~maskExtended], lw=0, marker='o')
    ax1.set_xlabel("MAG_APER(1)")
    ax1.set_ylabel("CLASS_STAR")
    
    ax2.scatter(catalog['MAG_APER(2)'][maskExtended], catalog['MAG_APER(2)'][maskExtended] - catalog['MAG_APER(3)'][maskExtended], c=catalog['CLASS_STAR'][maskExtended], lw=0, marker='>')
    ax2.scatter(catalog['MAG_APER(2)'][~maskExtended], catalog['MAG_APER(2)'][~maskExtended] - catalog['MAG_APER(3)'][~maskExtended], alpha=0.02, c=catalog['CLASS_STAR'][~maskExtended], lw=0, marker='o')
    ax2.set_xlabel("MAG_APER(2)")
    ax2.set_ylabel("MAG_APER(2) - MAG_APER(3)")
    
    ax3.scatter(catalog['MAG_APER(1)'][maskExtended], catalog['MAG_APER(1)'][maskExtended] - catalog['MAG_APER(2)'][maskExtended], c=catalog['CLASS_STAR'][maskExtended], lw=0, marker='>')
    ax3.scatter(catalog['MAG_APER(1)'][~maskExtended], catalog['MAG_APER(1)'][~maskExtended] - catalog['MAG_APER(2)'][~maskExtended], alpha=0.02, c=catalog['CLASS_STAR'][~maskExtended], lw=0, marker='o')
    ax3.set_xlabel("MAG_APER(1)")
    ax3.set_ylabel("MAG_APER(1) - MAG_APER(2)")
    
    ax4.scatter(catalog['MAG_APER(1)'][maskExtended], catalog['FWHM_IMAGE'][maskExtended], c=catalog['CLASS_STAR'][maskExtended], lw=0, marker='>')
    ax4.scatter(catalog['MAG_APER(1)'][~maskExtended], catalog['FWHM_IMAGE'][~maskExtended], alpha=0.02, c=catalog['CLASS_STAR'][~maskExtended], lw=0, marker='o')
    ax4.set_xlabel("MAG_APER(1)")
    ax4.set_ylabel("FWHM_IMAGE")
    ax4.set_ylim(0,10)
    '''
    
    plt.tight_layout()
    plt.show()
    #ax3.scatter(trans_data[0], trans_data[1])
    
def trimCatalog(catalog, imageOriginal, maskBad, sex, border=60, expand=3, magLimit=30):
    debug("Trimming catalog")
    maskBad = maskBad * 1
    maskBad[:,0:border] = 1
    maskBad[:, -border:] = 1
    maskBad[0:border, :] = 1
    maskBad[-border:,:] = 1
    expandConvolve = np.ones((5,5))
    
    ratio = 0.25
    maskShrunk = resize(maskBad, np.floor(np.array(maskBad.shape) * ratio), order=1, preserve_range=True)
    
    debug("Expanding bad pixel mask")
    for i in range(expand):
        maskShrunk = scipy.signal.convolve2d(maskShrunk, expandConvolve, mode="same")
        
    maskBad = resize(maskShrunk, maskBad.shape, order=1, preserve_range=True) #scipy.ndimage.interpolation.zoom        
    maskGood = (maskBad == 0)
    
    catalogTrimmed = catalog[(maskGood[catalog['Y_IMAGE'].astype(int) - 1,catalog['X_IMAGE'].astype(int) - 1])]
    
    debug("Removing sources that are too faint")
    for i in range(len(sex.config['PHOT_APERTURES'])):
        catalogTrimmed = catalogTrimmed[catalogTrimmed['MAG_APER(%d)'%(i+1)] < magLimit]
    if debugPlot:
        fig, ax0, ax1 = getDefaultImgComparison()
        ax0.set_title("Input image and catalog")
        ax1.set_title("Trimmed")
        im0 = ax0.imshow(imageOriginal, vmin=1310, vmax=1450, cmap=cmap, origin="lower")
        ax0.scatter(catalog['X_IMAGE'], catalog['Y_IMAGE'], color='r', s=1)
        im1 = ax1.imshow(imageOriginal, vmin=1310, vmax=1450, cmap=cmap, origin="lower")
        ax1.scatter(catalogTrimmed['X_IMAGE'], catalogTrimmed['Y_IMAGE'], color='r', s=1)

        addColourBars([(im0, ax0), (im1, ax1)])
        plt.show()
    debug("Trimming went from %d objects to %d objects" % (catalog.size, catalogTrimmed.size))
    return catalogTrimmed
    
def getCandidates(catalog, pixelThreshold=1):
    debug("Loading in Ricardo's extendeds")
    coord = np.loadtxt("../resources/candidates")
    mask = []
    for entry in catalog:
        dists = np.sqrt((entry['X_IMAGE'] - coord[:,0])**2 + (entry['Y_IMAGE'] - coord[:,1])**2 )
        mask.append(dists.min() < pixelThreshold)
    mask = np.array(mask)
    return mask
    
def normaliseRadial(catalog, sex, maskExtended):
    
    magArray = []
    debug("Constructing magnitude arrays")
    for entry in catalog:
        mags = [entry['MAG_APER(%d)'%(i+1)] for i in range(len(sex.config['PHOT_APERTURES']))]
        magArray.append(mags)
    magArray = np.array(magArray)
    magArrayNormalised = magArray - np.max(magArray, axis=1)[np.newaxis].T
    
    mins = np.min(magArrayNormalised, axis=1)
    magArrayNormalised = magArrayNormalised / mins[np.newaxis].T
    radii = sex.config['PHOT_APERTURES']
    
    result = []
    t = 1/np.sqrt(2)
    for row in magArrayNormalised:
        pixels = interp1d(row, radii, kind="linear")(t)
        result.append(np.round(pixels,3))
    result = np.array(result)
    
    debug("Removing 'MAG_APER' from catalog, replacing with 'FALLOUT'")
    names = [n for n in catalog.dtype.names if "MAG_APER" not in n]
    catalogFinal = catalogTrimmed[names].copy()
    catalogFinal = append_fields(catalogFinal, 'FALLOFF', result, usemask=False)    
    
    
    if debugPlot:
        debug("Creating radial plot")    
        fig = plt.figure(figsize=(15,5))
        ax0 = fig.add_subplot(1,3,1)
        ax1 = fig.add_subplot(1,3,2)
        ax2 = fig.add_subplot(1,3,3)
        for i,row in enumerate(magArrayNormalised):
            if maskExtended[i]:
                alpha = 1
                c = 'r'
            else:
                if np.random.random() > 0.9:
                    alpha = 0.05
                    c = 'b'
                else:
                    continue
            ax0.plot(radii, row, color=c, alpha=alpha)
            
        d = np.diff(radii)
        bins = np.concatenate(([radii[0] - d[0]], radii[1:] + 0.5*d))
        h1, ed1 = np.histogram(result[maskExtended], bins=bins, density=False)
        x1 = 0.5*(ed1[1:] + ed1[:-1])

        h2, ed2 = np.histogram(result[~maskExtended], bins=bins, density=False)
        x2 = 0.5*(ed2[1:] + ed2[:-1])
        width= 0.7 * diff(bins)
        ax1.bar(x1, 1.0*h1/h1.sum(), width=width, label="Extended", color='r', alpha=0.5)
        ax1.bar(x2, 1.0*h2/h2.sum(), width=width, label="Point", color='b', alpha=0.5)
        ax2.plot(x1, h1, label="Extended", color='r')
        ax2.plot(x2, h2, label="Point", color='b')
        ax2.set_yscale('log')
        ax1.set_xlabel("Pix at 0.707 normalised magnitude (Prob)")
        ax2.set_xlabel("Pix at 0.707 normalised magnitude (Number)")
        ax0.set_xlabel("Pix")
        ax0.set_ylabel("Normalised magnitude falloff")
        ax1.set_xlim(2, 8)
        ax1.legend()
        ax2.legend()
        plt.show()

    
    return catalogFinal
    
def getClassifier(catalog, mask, strength=15):
    #mask2 = catalog['CLASS_STAR'] < 0.5
    #y = (mask & mask2) * 1
    y = mask * 1
    X = catalog.view(np.float64).reshape(catalog.shape + (-1,))[:, 3:]
    
    debug("Creating classifier")
    # Create and fit an AdaBoosted decision tree
    bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=2), algorithm="SAMME", n_estimators=strength)
    bdt.fit(X, y, sample_weight=1+40*y)
    debug("Classifier created")
    bdt.label = "Boosted Decision Tree"
    return bdt
    
def getGC(classifier, catalog, ellipticity=0.25):
    X = catalog.view(np.float64).reshape(catalog.shape + (-1,))[:, 3:]
    z = classifier.predict(X) == 1
    gcs = z & (catalog['ELLIPTICITY'] < ellipticity)
    gal = z & (catalog['ELLIPTICITY'] > ellipticity)
    galaxies = (catalog[gal][['X_IMAGE', 'Y_IMAGE']])
    np.savetxt("../../../galaxies.txt", galaxies)
    return gcs
    
def testSelfClassify(classifier, catalog, y):
    X = catalog.view(np.float64).reshape(catalog.shape + (-1,))[:, 3:]
    debug("Testing classifier on training data for consistency")
    z = classifier.predict(X)
    
    correctPoint = (z == 0) & (~y)
    correctExtended = (z == 1) & (y)
    incorrectPoint = (z == 1) & (~y)
    incorrectExtended = (z == 0) & (y)
    
    debug("Extended correct: %0.1f%%" % (100.0 * correctExtended.sum() / y.sum()))
    debug("False negative: %0.1f%%" % (100.0 * incorrectExtended.sum() / (y).sum()))
    debug("Point correct: %0.1f%%" % (100.0 * correctPoint.sum() / (~y).sum()))
    debug("False positive: %0.1f%%" % (100.0 * incorrectPoint.sum() / (~y).sum()))
    
    
    
    if debugPlot:
        fig = plt.figure(figsize=(8,8))
        ax0 = fig.add_subplot(1,1,1)
        ax0.scatter(catalog[correctPoint]['FALLOFF'], catalog[correctPoint]['CLASS_STAR'], label="Correct Point", c="k", lw=0, alpha=0.1)
        ax0.scatter(catalog[correctExtended]['FALLOFF'], catalog[correctExtended]['CLASS_STAR'], label="Correct Extended", c="g", lw=0, s=30)
        ax0.scatter(catalog[incorrectExtended]['FALLOFF'], catalog[incorrectExtended]['CLASS_STAR'], label="False negative", c="r", marker="+", lw=1, s=30)
        ax0.scatter(catalog[incorrectPoint]['FALLOFF'], catalog[incorrectPoint]['CLASS_STAR'], label="False positive", c="m", lw=1, marker="x", s=20)
        ax0.legend()
        ax0.set_xlabel("FALLOFF")
        ax0.set_ylabel("CLASS_STAR")
        ax0.set_title(classifier.label)
        plt.show()
        
    return getGC(classifier, catalog)

def showInDS9(images, catalog=None, cols=['X_IMAGE','Y_IMAGE']):
    try:
        tempDir = tempfile.mkdtemp()
        debug("Generating temp directory at %s" % tempDir)
        
        fitsFile = fits.open(fitsPath)
        names = [tempDir + os.sep + "temp%d.fits"%i for i in range(len(images))]
        for i,(name,image) in enumerate(zip(names,images)):
            if isinstance(image, str):
                names[i] = os.path.abspath(image)
                images[i] = fits.getdata(image).astype(np.float32)
            else:
                tempFits = name
                fitsFile[0].data = image
                fitsFile.writeto(tempFits)
        fitsFile.close()
        
        commandline = "ds9 "
        commandline += " ".join(names)
        median = np.median(images[0])
        commandline += " -scale limits %d %d" % (median, median + 50)
        colours = ["cyan", "red", "green", "cyan"]
        if catalog is not None:
            cat = "catalog.txt"
            catFile = tempDir + os.sep + cat
            np.savetxt(catFile, catalog[cols], fmt="%0.1f")
            commandline += " -catalog import tsv %s -catalog psky image -catalog symbol shape circle -catalog symbol size 5 -catalog symbol size2 5 -catalog symbol color %s -catalog update " % (catFile, colours[0])
        
        f = "toRun.sh"
        filename = tempDir + os.sep + f
        with open(filename, 'w') as fil:
            fil.write(commandline + " &\n")
            
        st = os.stat(filename)
        os.chmod(filename, st.st_mode | stat.S_IEXEC)
        print("\n" + filename)
        
        #pid = subprocess.call(['/bin/bash', '-i', '-c', filename], stdout=subprocess.PIPE, stderr=subprocess.PIPE)    
        #outs = pid.communicate()

        raw_input("Press enter to close")
        return None

    except:
        raise
    finally:
        try:
            shutil.rmtree(tempDir)  # delete directory
            pass
        except OSError as exc:
            if exc.errno != errno.ENOENT:
                raise  # re-raise exception
                
                
def getPSFStars(catalog, sex, skyFlux, image, check=3):
    debug("Getting PSF stars")

    debug("\tEnforcing maximum flux thresholds")
    psfMask = (catalog['FLUX_MAX'] > 5 * skyFlux) & (catalog['FLUX_MAX'] < 0.8 * sex.config['SATUR_LEVEL'])    

    debug("\tEnforcing CLASS_STAR")
    psfMask = psfMask & (catalog['CLASS_STAR'] > 0.8)
    
    debug("\tEnforcing ELLIPTICITY")
    psfMask = psfMask & (catalog['ELLIPTICITY'] < 0.1)
    
    debug("\tEnforcing minimum distance from each source")
    for i,row in enumerate(catalog):
        if psfMask[i]:
            dists = np.sqrt((catalog['X_IMAGE'] - row['X_IMAGE'])**2 + (catalog['Y_IMAGE'] - row['Y_IMAGE'])**2)
            minDist = np.min(dists[dists > 1e-6])
            psfMask[i] = minDist > (row['FWHM_IMAGE'] * 3 + 15)
            
            
    
    debug("\tEnforcing no overflow")
    threshold = sex.config['SATUR_LEVEL']
    for i,row in enumerate(catalog):
        if psfMask[i]:
            x = np.round(row['X_IMAGE']).astype(np.int)
            y = np.round(row['Y_IMAGE']).astype(np.int)
            psfMask[i] = psfMask[i] and np.all(image[y-check:y+check, x-check:x+check] < threshold)
            psfMask[i] = psfMask[i] and np.all(image[y-1:y+1, x-1:x+1] > 4 * skyFlux)
            psfMask[i] = psfMask[i] and np.all(image[y-check:y+check, x-check:x+check] > 0)
            
    ysplit = image.shape[0] / 2;
    psfMask1 = psfMask & (catalog['Y_IMAGE'] > ysplit)
    psfMask2 = psfMask & (catalog['Y_IMAGE'] < ysplit)
    
            
    debug("\tReturning mask for %d,%d candidate masks" % (psfMask1.sum(), psfMask2.sum()))
    return [psfMask1, psfMask2]

def getPSF(fitsPath, catalog, psfs, scampFits):
    debug("Generating PSFs")
    tempDir = os.path.dirname(scampFits)
    tempDir = tempDir + os.sep + "psfs"
    try:
        shutil.rmtree(tempDir)  # delete directory
    except OSError as exc:
        if exc.errno != errno.ENOENT:
            raise  # re-raise exception
    os.mkdir(tempDir)
    # Copy image
    debug("\tCopying image fits to temp directory")
    shutil.copy(fitsPath, tempDir + os.sep + "imgPsf.fits")

    # Save psf locations
    debug("\tCreating PSF star position lists")
    for i,psf in enumerate(psfs):
        np.savetxt(tempDir + os.sep + "%d.cat"%i, catalog[psf][['X_IMAGE', 'Y_IMAGE']], fmt="%0.2f")


    # Update environ
    env = os.environ.copy()
    env['PATH'] = ("%s/variants/common/bin:%s/bin:%s/python/bin:" % (urekaPath, urekaPath, urekaPath))+env['PATH']
    env['PYTHONEXECUTABLE'] = "%s/python/bin/python" % urekaPath
    # Save script
    script = '''cd %s
daophot
digiphot
phot imgPsf %s output=default scale=1 fwhmpsf=3  sigma=3.5 readnoi=10 gain=GAIN calgori=centroid cbox=5 salgori=mode annulus=25 dannulus=30  aperture=5,8,10,15 zmag=25 interactive=no verify-
pstselect imgPsf photfile=default pstfile=default maxnpsf=300 psfrad=17 fitrad=3 verify-
psf imgPsf photfile=default pstfile=default psfimage=default opstfile=default groupfile=default scale=1 fwhmpsf=3  sigma=3.5 readnoi=10 gain=GAIN function=gauss varorder=1 saturate=no nclean=0 psfrad=17 fitrad=3 interac=no verify-
task als2bl = %s
als2bl imgPsf.psf.1.fits %s

.exit''' % (tempDir, "%s", baolabScript, "%s")


    # Run script for each psf
    psfNames = ["psf%d.fits"%i for i in range(len(psfs))]
    
    debug("\tSaving script and executing Pyraf")
    for (i,psf),name in zip(enumerate(psfs), psfNames):
        filename = tempDir + os.sep + "script.cl"
        with open(filename, 'w') as f:
            f.write(script % ("%d.cat"%i, name))
        debug("\tExecuting pyraf commands for psf catalog %d"%i)
            
        # Run script
        p = subprocess.Popen(["/bin/bash", "-i", "-c", "pyraf -x -s < script.cl && cp %s .."%name], env=env, stderr=subprocess.PIPE, stdout=subprocess.PIPE, cwd=tempDir)   
        output = p.communicate() #now wait
        #debug(output[0])
        #debug(output[1])
        
    realTempDir = os.path.dirname(scampFits)
    for n in psfNames:
        name = realTempDir + os.sep + n
        if not os.path.isfile(name):
            raise "Expected file %s not found" % name
        
    shutil.rmtree(tempDir)
    
    return psfNames
    


def runIshape(image, fitsPath, catalog, psfs, scampFits):
    debug("Running ishape on given image")
    tempDir = os.path.dirname(scampFits)
    tempDir = tempDir + os.sep + "ishape"
    debug("\tGenerating temp directory at %s" % tempDir)
    
    models = ["KING30", "SERSICx"]
    indexes = [30.0, 2.0]
    
    
    try:
        shutil.rmtree(tempDir)  # delete directory
    except OSError as exc:
        if exc.errno != errno.ENOENT:
            raise  # re-raise exception
    os.mkdir(tempDir)
    
    fitsFile = fits.open(scampFits)
    tempFits = tempDir + os.sep + "ishape.fits"
    fitsFile[0].data = image.astype(np.float32)
    debug("\tWriting image fits")
    fitsFile.writeto(tempFits, clobber=True)
    #fitsFile.close()
    

    ysplit = image.shape[0] / 2;
    psfMask1 = (catalog['Y_IMAGE'] > ysplit)
    psfMask2 = (catalog['Y_IMAGE'] < ysplit)

    tempCoord = tempDir + os.sep + "coords%d.txt"
    np.savetxt(tempCoord%0, catalog[psfMask1][['X_IMAGE', 'Y_IMAGE']], fmt="%d")
    np.savetxt(tempCoord%1, catalog[psfMask2][['X_IMAGE', 'Y_IMAGE']], fmt="%d")
    

    for i,psf in enumerate(psfs):
        for model,index in zip(models, indexes):
            argumentFile = tempDir + os.sep + "command.bl"
            with open(argumentFile, 'w') as f:
                f.write("cd %s\nishape %s %s %s LOGMODE=1 SHAPE=%s INDEX=%0.1f FITRAD=6 CALCERR=no\n" % (tempDir, "ishape.fits", "coords%d.txt"%i, '..' + os.sep + psf, model, index))
            
            commandline = 'bl < %s' % argumentFile
            
            #commandline = "bl < /Users/shinton/Downloads/bl/test.bl"
            debug("\tExecuting baolab and ishape for psf %s and model %s (%0.1f)" % (psf, model, index))
            p = subprocess.Popen(["/bin/bash", "-i", "-c", commandline], stderr=subprocess.PIPE, stdout=subprocess.PIPE)
            output = p.communicate() #now wait
            #print(output[0])
            #print(output[1])
    
    debug("\tReading log file")
    raw = []            
    with open(tempDir + os.sep + "ishape.log") as log:
        for line in log:
            if "@@F" in line or "@@E" in line:
                raw.append(line)
    parsed = {}
    # @#F <image>       <x> <y>   <psf>      <SHAPE> <FWHM> <RATIO> <PA> <CHISQR> <CHISQR0> <FLUX> <S/N> [INDEX] 
    # @#E <+/-FWHM> <+/-RATIO> <+/-PA> [+/-INDEX]
    x = ""
    y = ""
    shape = ""
    for r in raw:
        sep = r.split()
        if sep[0] == "@@F" and sep[6] != "OBJ-ERROR":
            x = sep[2]
            y = sep[3]
            shape = sep[5]
            fwhm = float(sep[6])
            ratio = float(sep[7])
            pa = float(sep[8])
            chi2 = float(sep[9])
            chi2d = float(sep[10])
            sn = float(sep[12])
            index = sep[13] if len(sep) >= 14 else ""
            shape += index
            key = x + " " + y
            if parsed.get(key) is None:
                parsed[key] = {}
            parsed[key][shape] = [fwhm, ratio, pa, chi2, chi2d, sn]
        if sep[0] == "@@E":
            parsed[key][shape] += [float(sep[1]), float(sep[2]), float(sep[3]), float(sep[4]), float(sep[5]), float(sep[6])]

    
    return parsed


    
    
def getScamped(fitsPath, image):
    # Create temp directory, put image in it, run sextractor
    debug("Creating polynomial distortion parameters using scamp")
    tempDir = tempfile.mkdtemp()
    debug("\tGenerating temp directory at %s" % tempDir)
    fitsFile = fits.open(fitsPath)
    tempFits = tempDir + os.sep + "image.fits"
    fitsFile[0].data = image.astype(np.float32)
    debug("\tWriting image fits")
    fitsFile.writeto(tempFits)
    fitsFile.close()
    
    # Use extractor output to run scamp
    sex = getScampSextractor()
    sex.config['CATALOG_NAME'] = "image.cat"
    debug('\tRunning sextractor for scamp')
    data = sex.run(tempFits, path=sexPath, clean=False, fileOutput=True, cwd=tempDir)
    print(data['output'])
    catalogName = tempDir + os.sep + data['filename']
    
    debug('\tInvoking scamp using output catalog')
    env = os.environ.copy()
    env['PATH'] = '/usr/local/bin:'+env['PATH']
    try:
        output = subprocess.check_output("/bin/bash -i -c \"" + scampPath + " " + data['filename'] + "\"", stderr=subprocess.STDOUT, shell=True, cwd=tempDir, env=env)
    except subprocess.CalledProcessError, e:
        print(e.output)
        
    
    debug('\tScamp output:\n')
    debug(output)
    
    debug('\tGetting .head file')
    headFile = tempDir + os.sep + "image.head"
    if (not os.path.exists(headFile)):
        raise Exception("Head file not found at " + headFile)
        
    debug("\tRunning missfits to incorporate scamp header")
    try:
        output2 = subprocess.check_output("/bin/bash -i -c \"" + missfitsPath + " image.fits\"", stderr=subprocess.STDOUT, shell=True, cwd=tempDir, env=env)
    except subprocess.CalledProcessError, e:
        print(e.output)
        
    debug("\tMissfits output:\n")
    debug(output)
    
    debug("\tReturning path to modified fits file")
    return tempFits
    # Return path to this fits file

def cleanDirectory(d):
    try:
        shutil.rmtree(d)  # delete directory
        pass
    except OSError as exc:
        if exc.errno != errno.ENOENT:
            raise  # re-raise exception
  


def refineGCMask(catalog, mask, parsed, maskExtended):
    debug("Refining GC mask using ishape results")
    newMask = mask.copy()
    extendeds = []
    notExtendeds = []
    result = []
    for i,row in enumerate(catalog):
        if newMask[i]:        
            x = row['X_IMAGE']
            y = row['Y_IMAGE']
            key = "%d %d" % (x,y)
            #print(key)
            p = None
            if parsed.get(key) is not None:
                p = parsed[key]
            else:
                for i in range(-2,3):
                    for j in range(-2,3):
                        key = "%d %d" % (x+i,y+j)
                        if parsed.get(key) is not None:
                            p = parsed[key]
                            break
            if p is not None:
                chi2King = p['KING30'][3]
                chi2Delta = p['KING30'][4]
                chi2Sersic2 = p['SERSICx2.00'][3]
                result.append(p['KING30'] + p['SERSICx2.00'])
                if maskExtended[i]:
                    extendeds.append(chi2King/chi2Delta)
                    print("%d"%x,"%d"%y,chi2King, chi2Delta, chi2Sersic2, chi2King/chi2Delta)
                else:
                    notExtendeds.append(chi2King/chi2Delta)

                newMask[i] &= chi2King < 0.75 * chi2Delta
            else:
                newMask[i] = False
                
    if debugPlot:
        fig = plt.figure()
        ax0 = fig.add_subplot(1,1,1)
        extendeds = np.array(extendeds)
        notExtendeds = np.array(notExtendeds)
        ax0.hist(extendeds, normed=1, color='r', alpha=0.5, label="Marked extended")
        ax0.hist(notExtendeds, normed=1, alpha=0.5, label="Not marked extended")
        ax0.legend(loc=2)
        ax0.set_xlabel(r"$\chi^2_{\rm KING30} / \chi^2_\delta$", fontsize=18)
        ax0.set_ylabel("Prob")
    
    debug("\tNew mask has %d gcs, compared to %d previously" % (newMask.sum(), mask.sum()))
    return newMask, np.array(result)
        
        
def xy2sky(scampFits, fitsFile, positions, mosaics):
    debug("\tRunning xy2sky")
    tempDir = os.path.dirname(scampFits)
    tempDir = tempDir + os.sep + "xy2sky"
    debug("\tGenerating temp directory at %s" % tempDir)
    
    try:
        shutil.rmtree(tempDir)  # delete directory
    except OSError as exc:
        if exc.errno != errno.ENOENT:
            raise  # re-raise exception
    os.mkdir(tempDir)
    
    imfile = tempDir + os.sep + "imfile.txt"
    outfile = tempDir + os.sep + "skys.txt"
    np.savetxt(imfile, positions, fmt="%0.3f")
    
    commandline = wcsToolsPath + "/xy2sky %s @%s | cut -d \" \" -f 1-2 > %s" % (fitsFile, imfile, outfile)
    p = subprocess.Popen(["/bin/bash", "-i", "-c", commandline], stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    output = p.communicate() #now wait
    
    coords = []
    for mosaic in mosaics:
        tempOut = tempDir + os.sep + "temp.txt"
        commandline = wcsToolsPath + "/sky2xy %s @%s | cut -d \" \" -f 6,8 > %s" % (mosaic, outfile, tempOut)
        p = subprocess.Popen(["/bin/bash", "-i", "-c", commandline], stderr=subprocess.PIPE, stdout=subprocess.PIPE)
        output = p.communicate() #now wait
        debug("\t%s" % output[0])
        debug("\t%s" % output[1])
        xys = np.loadtxt(tempOut)
        coords.append(xys)
    return coords
    
    
def getCatalogPixelsInMosaic(scampFits, catalog, mosaicFileI, mosaicFileR, mosaicFileZ):
    
    mosaicPixels = xy2sky(scampFits, scampFits, catalog[['X_IMAGE', 'Y_IMAGE']], [mosaicFileI, mosaicFileR, mosaicFileZ])
    catalogMosaic = append_fields(catalog,  'I_X', mosaicPixels[0][:,0], usemask=False)
    catalogMosaic = append_fields(catalogMosaic, 'I_Y', mosaicPixels[0][:,1], usemask=False)    
    catalogMosaic = append_fields(catalogMosaic, 'R_X', mosaicPixels[1][:,0], usemask=False)    
    catalogMosaic = append_fields(catalogMosaic, 'R_Y', mosaicPixels[1][:,1], usemask=False)    
    catalogMosaic = append_fields(catalogMosaic, 'Z_X', mosaicPixels[2][:,0], usemask=False)    
    catalogMosaic = append_fields(catalogMosaic, 'Z_Y', mosaicPixels[2][:,1], usemask=False)    

    return catalogMosaic




  
## Get the fits file
'''
fitsFile = fits.open(fitsPath)
imageOriginal = fitsFile[0].data
fitsFile.close()

## Add mask to help out the median convolution
mask, imageMasked = getMaskedImage(imageOriginal)

## Perform the median convolution to remove galaxy light pollution
(outputName, imageBackground) = broadClean(imageMasked, redo=False, fast=True)

## Subtract out the light pollution
imageSubtracted = getSubtracted(imageOriginal, imageBackground, fitsPath)

## Find the correct sky flux
skyFlux, imageSky = addSkyFlux(imageOriginal, imageSubtracted)

scampFits = getScamped(fitsPath, imageSubtracted)

## Get the object catalogs from sextractor
#catalog, sex = getCatalogs(scampFits, imageSubtracted)



## Trim Catalogs
catalogTrimmed = trimCatalog(catalog, imageOriginal, mask, sex)




## Get mask for estimated extendeds
maskExtended = getCandidates(catalogTrimmed)

## Manipulate catalog
catalogFinal = normaliseRadial(catalogTrimmed, sex, maskExtended)

# Plot some statistics for the objects
#visualiseHigherDimensions(catalogFinal, maskExtended)
#checkForClustering(catalogTrimmed)
#showStats(catalogFinal, maskExtended)
#'''


'''
classifier = getClassifier(catalogFinal, maskExtended)


gcsMask = testSelfClassify(classifier, catalogFinal, maskExtended)

#'''
'''

psfMask1, psfMask2 = getPSFStars(catalogFinal, sex, skyFlux, imageOriginal)
psfFitsFiles = getPSF(fitsPath, catalogFinal, [psfMask1, psfMask2], scampFits)


#substractPSF(imageSky, catalogFinal, prf, gcsMask)
#pid = showInDS9([imageOriginal], catalog=catalogFinal[psfMask])

'''
'''
parsedIshape = runIshape(imageSky, fitsPath, catalogFinal[gcsMask], psfFitsFiles, scampFits)

gcsMask2, ishapeRes = refineGCMask(catalogFinal, gcsMask, parsedIshape, maskExtended)

#'''
#catalogMosaic = getCatalogPixelsInMosaic(scampFits, catalogFinal[gcsMask2], mosaicFileI, mosaicFileR, mosaicFileZ)

showInDS9([mosaicFileZ], catalog=catalogMosaic, cols=['Z_X','Z_Y'])
#cleanDirectory(os.path.dirname(scampFits))


