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
        
        #shutil.copy(fitsFile, tempDir + os.sep + "temp.fits")
        commandline = "ds9 "
        commandline += " %s "%os.path.abspath(fitsFile)
        median = np.median(fits.getdata(fitsFile))
        commandline += " -scale limits %d %d" % (median - 100, median + 300)
        colours = ["red", "green", "cyan"]
        if catalog is not None:
            cat = "catalog.txt"
            catFile = tempDir + os.sep + cat
            np.savetxt(catFile, catalog[cols], fmt="%0.5f")
            commandline += " -catalog import tsv %s %s -catalog symbol shape circle -catalog symbol size 15 -catalog symbol size2 15 -catalog symbol color %s -catalog update " % (catFile, ("-catalog psky fk5 -catalog psystem wcs" if cols[0]=="RA" else "-catalog psky image"), colours[0])
        
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

def removeDoubles(catalog, threshold=4):
    print("Removing duplicates. Starting with %d entries"%catalog.shape[0])
    newCat = catalog.copy()
    for i,row in enumerate(catalog):
        x = row['X_IMAGE']
        y = row['Y_IMAGE']
        dists = np.sqrt((newCat['X_IMAGE']-x)*(newCat['X_IMAGE']-x) + (newCat['Y_IMAGE']-y)*(newCat['Y_IMAGE']-y))
        dists[dists == 0] = 99
        minDist = dists.argmin()
        if (dists[minDist] < threshold):
            newCat = np.delete(newCat, (minDist), axis=0)
    print("Revmoed duplicates. Ending with %d entries"%newCat.shape[0])

    return newCat
    
    
def latexPrint(catalog,label, columns=['RA','DEC','ELLIPTICITY','Z_MAG','RMZ_11'], labels=["RA","DEC",r"$\epsilon$","$z'$","$r'-z'$"], positions=["l","l","c","c","c"], formats=["%s","%s", "%0.2f","%0.3f","%0.3f"]):
    from astropy import units as u
    from astropy.coordinates import SkyCoord
    
    
    positions = positions
    columns = columns
    string = "\\begin{tabular}{l" + "".join(positions) + "}\n\\hline\n"
    string += "ID "
    for column in labels:
        string += "& %s "%column
    string += "\\\\\n\hline\n"
    for i,row in enumerate(catalog):
        c = SkyCoord(ra=row['RA']*u.degree, dec=row['DEC']*u.degree) 
        rahms = c.ra.hms
        string += "%s%d "%(label,i+1)
        for f,column in zip(formats,columns):
            string += ("& %s "%f)%row[column]
        string += "\\\\\n"
    string += "\\hline\n\\end{tabular}"
    return string
        
def getZcal(z):
    pz = -2.082443#-2.193601 #
    return z - pz
    
def getRcal(r, z):
    pr = -2.697975
    return r - pr
    
def getIcal(i):
    pi = -2.61702 #-2.717 #
    return i - pi
    
def correctRAirMass(airmass):
    ext = getAtmosphericAbsorption(6400)
    return ext * airmass
    
def correctIAirMass(airmass):
    ext = getAtmosphericAbsorption(7500)
    return ext * airmass

def correctZAirMass(airmass):
    ext = getAtmosphericAbsorption(9440)
    return ext * airmass    
    
def getAtmosphericAbsorption(angstroms):
    x = [3100, 3200, 3400, 3600, 3800, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 8000, 9000, 12500 ]
    y = [1.37, 0.82, 0.51, 0.37, 0.30, 0.25, 0.17, 0.13, 0.12, 0.11, 0.11, 0.10, 0.07, 0.05, 0.015]
    f = interp1d(x,y,kind="linear")
    return f(angstroms)


def addColourDiff(catalog):
    cat = catalog.copy()
    apertures = [int(n[n.find("_")+1:]) for n in cat.dtype.names if n.find("R_") != -1]
    
    for ap in apertures:
        z = "Z_%d"%ap
        r = "R_%d"%ap
        diff = cat[r] - cat[z]
        cat = append_fields(cat, 'RMZ_%d'%ap, diff, usemask=False)    
    return cat
    
def plotColourDifference(cat, psfs, number=20, threshold=3):
    

    
    cols = [n for n in cat.dtype.names if n.find("RMZ_") != -1]
    entries = cat[cols].view(np.float64).reshape(cat.shape + (-1,))
    entries2 = psfs[cols].view(np.float64).reshape(psfs.shape + (-1,))
    aps = np.array([int(n[n.find("_")+1:]) for n in cols])
    
    adj2 = entries2 - np.min(entries2, axis=1)[np.newaxis].T
    medians2 = np.median(adj2, axis=0)
    
    adj = entries - np.min(entries, axis=1)[np.newaxis].T
    
    dev = np.max(adj2, axis=1)
    minDev = np.median(dev)
    print(minDev)
    medians = np.median(adj, axis=0)
    
    
    fig, ax0 = plt.subplots(figsize=(7,5), ncols=1, sharey=True)
    ax0.invert_yaxis()
    ax0.plot(aps, medians, 'b', label="Median GC candidate")
    ax0.plot(aps, medians2, 'r', label="Median star candidate")
    ax0.legend(loc=4)
    ax0.set_xlabel("Aparture size (px)")
    ax0.set_ylabel("$r' - z'$", fontsize=16)
    
    fig, ax0 = plt.subplots(figsize=(7,5), ncols=1, sharey=True)
    ax0.invert_yaxis()
    for i,row in enumerate(entries2):
        mask = np.isfinite(row) & (row < 30) & (row > -10)
        red = row[mask]
        if i < number:
            ax0.plot(aps[mask], row[mask], 'r', alpha=0.3)    
    
    
    gcMask = []
    for i,row in enumerate(entries):
        mask = np.isfinite(row) & (row < 30)
        red = row[mask]
        if red.size > 0:
            gcMask.append(red.max() - red.min() < 5 * minDev)
        else:
            gcMask.append(False)
        if i < number:
            ax0.plot(aps[mask], row[mask], 'b', alpha=0.2)

            
            
    ax0.set_xlabel("Aparture size (px)")
    ax0.set_ylabel("$r' - z'$", fontsize=16)
    return cat[np.array(gcMask)]
        
                
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
    vmax = 3    
    
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
    fig.savefig("colour.pdf", bbox_inches="tight")