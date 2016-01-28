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


def showInDS9(fitsFile, catalog=None, cols=['X_IMAGE','Y_IMAGE']):
    try:
        tempDir = tempfile.mkdtemp()
        print("Generating temp directory at %s" % tempDir)
        
        shutil.copy(fitsFile, tempDir + os.sep + "temp.fits")
        commandline = "ds9 "
        commandline += " %s "%fitsFile
        median = np.median(fits.getdata(fitsFile))
        commandline += " -scale limits %d %d" % (median, median + 50)
        colours = ["red", "green", "cyan"]
        if catalog is not None:
            cat = "catalog.txt"
            catFile = tempDir + os.sep + cat
            np.savetxt(catFile, catalog[cols], fmt="%0.5f")
            commandline += " -catalog import tsv %s -catalog psky image -catalog symbol shape circle -catalog symbol size 10 -catalog symbol size2 10 -catalog symbol color %s -catalog update " % (catFile, colours[0])
        
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
                
                
                
def plotColourDiagrams(cat, colourColumn='Chi2DeltaKingDiv'):
    mag = cat['Z_MAG']
    z = cat['Z_8']
    i = cat['I_8']
    r = cat['R_8']
    
    zmask = cat['Z_MASK'] & (cat['Z_8'] < 90)
    imask = cat['Z_MASK'] & (cat['I_8'] < 90)
    rmask = cat['Z_MASK'] & (cat['R_8'] < 90)
    
    
    
    fig, axes = plt.subplots(figsize=(15,5), ncols=3, sharey=True)
    axes[0].invert_yaxis()
    
    # ELLIPTICITY gives a lot of variation. However max is 0.25, so still in possible GC range
    # FLUX_AUTO shows potential truncation on left hand side of plots
    # CI shows some outliers, but not as much as ELLIPTICITY
    # CI2 is much the same
    # KingFWHM has a few massive outliers that should be removed
    # Chi2DeltaKingDiv has a small subset which kings are FAR better than delta. Class A candidates maybe. A lot hover around 1.0
    # FWHM_IMAGE has some very large FWHMs, maybe should remove those as well
    
    # FLIP AXIS ON Y BECAUSE HA (i-z, r-i, r-z)
    # FIND DUST MAP FOR OUR FOV (whether data or webserver)
    
    
    cmap = 'viridis'
    vmin = cat[zmask | imask | rmask][colourColumn].min()
    vmax = cat[zmask | imask | rmask][colourColumn].max()
    #vmax = 3    
    
    h1 = axes[0].scatter(i[zmask & imask] - z[zmask & imask], mag[zmask & imask], c=cat[zmask & imask][colourColumn], vmin=vmin, vmax=vmax, edgecolor="none", cmap=cmap)
    h2 = axes[1].scatter(r[rmask & imask] - i[rmask & imask], mag[rmask & imask], c=cat[rmask & imask][colourColumn], vmin=vmin, vmax=vmax, edgecolor="none", cmap=cmap)
    h3 = axes[2].scatter(r[zmask & rmask] - z[zmask & rmask], mag[zmask & rmask], c=cat[zmask & rmask][colourColumn], vmin=vmin, vmax=vmax, edgecolor="none", cmap=cmap)
    
    divider = make_axes_locatable(axes[2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    #cbaxes = fig.add_axes([0.905, 0.13, 0.01, 0.77])
    cb = plt.colorbar(h1, cax = cax)  
    cb.set_label(colourColumn)
    
    axes[0].set_xlabel("$i' - z'$", fontsize=16)
    axes[1].set_xlabel("$r' - i'$", fontsize=16)
    axes[2].set_xlabel("$r' - z'$", fontsize=16)
    
    axes[0].set_ylabel("$z'$", fontsize=16)
    #axes[1].set_ylabel("$i'$", fontsize=16)
    #axes[2].set_ylabel("$z'$", fontsize=16)

    plt.tight_layout()