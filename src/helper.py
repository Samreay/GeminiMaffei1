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
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.legend_handler import HandlerLine2D
from scipy.interpolate import interp1d


from matplotlib import ticker

from scipy.optimize import curve_fit


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

def removeDoubles(catalog, threshold=10):
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
    
def addFWHM(cat):
    as2Rad = np.pi / (180 * 60 * 60)
    pixels = cat['KingFWHM']
    parsecs = pixels / (0.2 / np.tan(2.7e6 * as2Rad)) 
    cat = append_fields(cat, 'KFWHM', parsecs, usemask=False)
    return cat

    
def latexPrint(catalog,label, columns=['RA','DEC','ELLIPTICITY','Z_ABS', 'Z_MAG','RMZ_11', 'KFWHM'], labels=["RA","DEC",r"$\epsilon$","$M_{z'}$","$m_{z'}$","$r'-z'$", 'King$_{30}$ FWHM (pc)'], positions=["l","l","c","c","c","c","c"], formats=["%s","%s", "%0.2f","%0.3f","%0.3f","%0.3f","%0.1f"]):
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
        string += "%s%d "%(label,i+1)
        for f,column in zip(formats,columns):
            val = row[column]
            if column == 'RA':
                val = r"$%d^{\rm{h}}\,%d^{\rm{m}}\,%0.1f^{\rm{s}}$"%c.ra.hms
            elif column == 'DEC':
                val = r"$%d^\circ\,%d'\,%0.2f''$"%c.dec.dms
            string += ("& %s "%f)%val
        string += "\\\\\n"
    string += "\\hline\n\\end{tabular}"
    return string
        
def getMZ():
    V = 6.47
    B = 7.35
    I = 5.41
    g = V + 0.60*(B-V) - 0.12
    r = V - 0.42*(B-V) + 0.11
    
    gg = 7.043
    rr = 6.244
    zz = 5.641
    ggmzz = gg - zz
    rrmzz = rr - zz
    
    z1 = gg - ggmzz
    z2 = rr - rrmzz
    res = np.mean(np.array([z1,z2]))
    print(res)
    return res
    
def getZcal(z):
    pz = -2.082443#-2.193601 #
    return z - pz
    
def getRcal(r):
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

mosaicZTime = 10#27
mosaicRTime = 10#75
mosaicITime = 10#27
def addColourDiff(catalog):
    cat = catalog.copy()
    apertures = [int(n[n.find("_")+1:]) for n in cat.dtype.names if n.find("R_") != -1]
    if 'Z_MAG' in cat.dtype.names:
        cat['Z_MAG'] += 2.5*np.log10(mosaicZTime)
        cat['Z_MAG'] = getZcal(cat['Z_MAG'])
        cat['Z_MAG'] += correctZAirMass(1.6434 - 1.31)

        cat = append_fields(cat, 'Z_ABS', getAbsolute(cat['Z_MAG']), usemask=False)    

    for ap in apertures:
        z = "Z_%d"%ap
        r = "R_%d"%ap
        i = "I_%d"%ap
        cat[z] = getZcal(cat[z])
        cat[z] += 2.5 * np.log10(mosaicZTime)
        cat[z] += correctZAirMass(1.6434 - 1.31)

        cat[r] = getRcal(cat[r])
        cat[r] += correctRAirMass(1.5659 - 1.303)
        cat[r] += 2.5 * np.log10(mosaicRTime)
        
        cat[i] = getIcal(cat[i])
        cat[i] += 2.5 * np.log10(mosaicITime)
        cat[i] += correctIAirMass(1.1082 - 1.32)


    
    for ap in apertures:
        z = "Z_%d"%ap
        r = "R_%d"%ap
        diff = cat[r] - cat[z]
        cat = append_fields(cat, 'RMZ_%d'%ap, diff, usemask=False)    
    return cat
    
    
def getDists(cat):
    mr = 39.1477583
    md = 59.654872
    
    dists = np.sqrt((cat['RA']-mr)**2 + (cat['DEC']-md)**2)
    dists *= 60
    cat = append_fields(cat, 'DIST', dists, usemask=False)
    return cat

def plotDist(cat, allS):
    bins = [0,2.5,5,7.5,10,15,20,25,30]
    h, e = np.histogram(cat['DIST'], bins=bins)
    hs, es = np.histogram(allS['DIST'], bins=bins)
    c = 0.5 * (e[:-1] + e[1:])
     
    
    d2 = 1.0 * h / hs
    d2 /= d2.max()
    #density = 1.0 * h / ((np.pi * e[1:] *e[1:]) - (np.pi * e[:-1] * e[:-1]))
    #d2 = density / density.max()
    
    fig, ax0 = plt.subplots(figsize=(5,4), ncols=1, sharey=True)
    #ax0.plot(c, 1.0*h/h.max(), 'b', label="By radius")
    ax0.plot(c, d2, 'b', label="Density ratio")
    ax0.plot(c, hs*1.0/hs.max(), 'r', label="Density all")
    ax0.plot(c, h*1.0/h.max(), 'g', label="Density all")
    #ax0.legend(loc=4)
    ax0.set_xlabel(r"$\rm{Arcminutes\ from\ Maffei\ 1\ center}$", fontsize=16)
    ax0.set_ylabel(r"$\rm{Relaitve\ area\ density}$", fontsize=16)
    plt.tight_layout()
    fig.savefig("density.pdf", bbox_inches="tight")
    
    
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
        
        
def getAbsolute(apparent, distance=2.7e6):
    return apparent - 2.5 * np.log10((distance/10)**2)
                
                
def plotSizeDiagrams(cat, classA):
    label=r"$\chi^2_{\rm{Delta}} / \chi^2_{\rm{King30}}$"
    fig, ax0 = plt.subplots(figsize=(5,4.5))
    ax0.invert_yaxis()
    
    y = cat['Z_ABS']
    x = cat['KFWHM']
    c = cat['Chi2DeltaKingDiv']
    mask = cat['Z_MASK']
    vmin = 1
    vmax = 3
    cmap = 'viridis'

    h1 = ax0.scatter(x[mask & ~classA], y[mask & ~classA], c=c[mask & ~classA],edgecolor="none", cmap=cmap,vmin=vmin, vmax=vmax, marker=">", label="Class B") # 
    h1 = ax0.scatter(x[mask & classA], y[mask & classA], c=c[mask & classA],edgecolor="none", cmap=cmap,vmin=vmin, vmax=vmax, label="Class A") # 
    ax0.set_ylabel(r"$M_{z'}$", fontsize=16)
    ax0.set_xlabel(r"$\rm{King30\ FWHM\ (pc)}$", fontsize=16)
    
    caa = mlines.Line2D([], [], color='#69CF37', markeredgecolor='#69CF37', linewidth=0, marker='o', linestyle='none', markersize=8, label='Class A')
    cbb = mlines.Line2D([], [], color='#1B737B', markeredgecolor='#1B737B', linewidth=0, marker='>', linestyle='none', markersize=8, label='Class B')
    ax0.legend(handler_map={caa: HandlerLine2D(numpoints=1), cbb: HandlerLine2D(numpoints=1)}, handles=[caa, cbb]) #)

    divider = make_axes_locatable(ax0)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cb = plt.colorbar(h1, cax = cax, ticks=[1, 1.5, 2,2.5, 3])  
    cb.set_label(label, fontsize=16)
    
    ax0.locator_params(nbins=6)
    ax0.axis('tight')
    #cax.locator_params(nbins=6)
    plt.tight_layout()
    fig.savefig("kingFWHM.pdf", bbox_inches="tight")
    fig.set_size_inches(3.5, 3)
    ax0.legend(handler_map={caa: HandlerLine2D(numpoints=1), cbb: HandlerLine2D(numpoints=1)}, frameon=False, borderpad=0, labelspacing=0, columnspacing=0, handles=[caa, cbb],bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.) #)
    fig.savefig("kingFWHM.png", transparent=True, bbox_inches="tight", dpi=300)
    
def plotSizeHistogram(cat, classA):

    data1 = cat[classA]['KFWHM']    
    data2 = cat[~classA]['KFWHM']
    bins = np.arange(0,15,1)+0.5
    binc = 0.5 * (bins[:-1] + bins[1:])
    
    h1,b = np.histogram(data1, bins=bins)
    h2,b = np.histogram(data2, bins=bins)
    
    fig,ax0 = plt.subplots(figsize=(5,4))
    
    ax0.bar(binc, h1, edgecolor='none', facecolor='#69CF37', label="Class A", align='center')
    ax0.bar(binc, h2, bottom=h1, edgecolor='none', facecolor='#1B737B', label="Class B", align='center')
    ax0.legend(loc=1)
    ax0.set_xlabel(r"$\rm{King30\ FWHM\ (pc)}$", fontsize=16)
    ax0.set_ylabel(r"$\rm{Count}$", fontsize=16)
    ax0.xaxis.set_ticks(binc)
    #ax0.axis('tight')
    #ax0.margins(0.05, 0.01)
    plt.tight_layout()
    #fig.savefig("sizeHist.pdf", bbox_inches="tight")
    fig.set_size_inches(3, 3)
    ax0.xaxis.set_ticks(binc[::2])

    fig.savefig("sizeHist.png", bbox_inches="tight", dpi=300, transparent=True)

def plotSizeHistogram2(cat, classA):

    data0 = cat['KFWHM']    
    data1 = cat[classA]['KFWHM']    
    data2 = cat[~classA]['KFWHM']
    
    bins0 = getOptimalBinSize(data0)
    bins1 = getOptimalBinSize(data1)
    bins2 = getOptimalBinSize(data2)
    bins1 = bins0 = bins2 = np.arange(0,15) + 0.5
    
    
    binc0 = 0.5 * (bins0[:-1] + bins0[1:])
    binc1 = 0.5 * (bins1[:-1] + bins1[1:])
    binc2 = 0.5 * (bins2[:-1] + bins2[1:])
    
    density = False
    h0,b = np.histogram(data0, bins=bins0, density=density)
    h1,b = np.histogram(data1, bins=bins1, density=density)
    h2,b = np.histogram(data2, bins=bins2, density=density)
    
    
    fig,ax0 = plt.subplots(figsize=(4.5,3.5))
    
    ax0.hist(binc1, bins=bins1, weights=h1, histtype='step', color='#69CF37', linewidth=2, label="Class A")
    ax0.hist(binc2, bins=bins2, weights=h2, histtype='step', color='#1B737B', ls="--", linewidth=2, label="Class B")
    ax0.hist(binc0, bins=bins0, weights=h0, histtype='step', color='k', ls=":", linewidth=2, label="Total")
    #ax0.bar(binc, h2, bottom=h1, edgecolor='none', facecolor='#1B737B', label="Class B", align='center')
    ax0.legend(loc=1)
    ax0.set_xlabel(r"$\rm{King30\ FWHM\ (pc)}$", fontsize=16)
    ax0.set_ylabel(r"$\rm{N}$", fontsize=16)
    ax0.xaxis.set_ticks(np.arange(1,15))
    #ax0.axis('tight')
    #ax0.margins(0.05, 0.01)
    plt.tight_layout()
    fig.savefig("sizeHist.pdf", bbox_inches="tight")

    #fig.savefig("sizeHist.png", bbox_inches="tight", dpi=300, transparent=True)
    
    
def plotColourHistogram(cat, classA):
    data0 = cat['RMZ_9']
    data1 = data0[classA]
    data2 = data0[~classA]    
    
    bins0 = getOptimalBinSize(data0)
    bins1 = getOptimalBinSize(data1)
    bins2 = getOptimalBinSize(data2)
    gap = 0.2
    bins1 = bins0 = bins2 = np.arange(-1,3.5,gap) + gap/2
    
    binc0 = 0.5 * (bins0[:-1] + bins0[1:])
    binc1 = 0.5 * (bins1[:-1] + bins1[1:])
    binc2 = 0.5 * (bins2[:-1] + bins2[1:])


    density = False
    h0,b = np.histogram(data0, bins=bins0, density=density)
    h1,b = np.histogram(data1, bins=bins1, density=density)
    h2,b = np.histogram(data2, bins=bins2, density=density)
    
    fig,ax0 = plt.subplots(figsize=(4.5,3.5))
    
    ax0.hist(binc1, bins=bins1, weights=h1, histtype='step', color='#69CF37', linewidth=2, label="Class A")
    ax0.hist(binc2, bins=bins2, weights=h2, histtype='step', color='#1B737B', ls="--", linewidth=2, label="Class B")
    ax0.hist(binc0, bins=bins0, weights=h0, histtype='step', color='k', ls=":", linewidth=2, label="Total")
    
    
    ax0.locator_params(nbins=6, axis='x')
    ax0.locator_params(nbins=8, axis='y')
    
    
    ax0.legend(loc=1)
    ax0.set_xlabel(r"$r' - z'$", fontsize=16)
    ax0.set_ylabel(r"$\rm{N}$", fontsize=16)
    #ax0.xaxis.set_ticks(np.arange(1,15))
    #ax0.axis('tight')
    #ax0.margins(0.05, 0.01)
    plt.tight_layout()
    fig.savefig("colourHist.pdf", bbox_inches="tight")
    fig.savefig("colourHist.png", bbox_inches="tight", transparent=True, dpi=300)

    #fig.savefig("sizeHist.png", bbox_inches="tight", dpi=300, transparent=True)
    
def plotColourDiagrams(cat, classA, colourColumn='Chi2DeltaKingDiv', label=r"$\chi^2_{\rm{Delta}} / \chi^2_{\rm{King30}}$"):
    mag = cat['Z_ABS']
    z = cat['Z_8']
    i = cat['I_8']
    r = cat['R_8']
    
    zmask = cat['Z_MASK'] & (cat['Z_8'] < 90)
    imask = cat['Z_MASK'] & (cat['I_8'] < 90)
    rmask = cat['Z_MASK'] & (cat['R_8'] < 90)
    
    
    vrmz = 0.43275675675675679
    vrmi = 0.28572500000000012
    vimz = 0.14756756756756753
    
    fig, axes = plt.subplots(figsize=(12,4), ncols=3, sharey=True)
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
    vmin = 1
    
    
    h1 = axes[0].scatter(i[zmask & imask & ~classA] - z[zmask & imask & ~classA], mag[zmask & imask & ~classA], c=cat[zmask & imask & ~classA][colourColumn], vmin=vmin, vmax=vmax, edgecolor="none", cmap=cmap, marker=">", label="Class B")
    h2 = axes[1].scatter(r[rmask & imask & ~classA] - i[rmask & imask & ~classA], mag[rmask & imask & ~classA], c=cat[rmask & imask & ~classA][colourColumn], vmin=vmin, vmax=vmax, edgecolor="none", cmap=cmap, marker=">")
    h3 = axes[2].scatter(r[zmask & rmask & ~classA] - z[zmask & rmask & ~classA], mag[zmask & rmask & ~classA], c=cat[zmask & rmask & ~classA][colourColumn], vmin=vmin, vmax=vmax, edgecolor="none", cmap=cmap, marker=">")
    h1 = axes[0].scatter(i[zmask & imask & classA] - z[zmask & imask & classA], mag[zmask & imask & classA], c=cat[zmask & imask & classA][colourColumn], vmin=vmin, vmax=vmax, edgecolor="none", cmap=cmap, label="Class A")
    h2 = axes[1].scatter(r[rmask & imask & classA] - i[rmask & imask & classA], mag[rmask & imask & classA], c=cat[rmask & imask & classA][colourColumn], vmin=vmin, vmax=vmax, edgecolor="none", cmap=cmap)
    h3 = axes[2].scatter(r[zmask & rmask & classA] - z[zmask & rmask & classA], mag[zmask & rmask & classA], c=cat[zmask & rmask & classA][colourColumn], vmin=vmin, vmax=vmax, edgecolor="none", cmap=cmap)

    print("Class A mean rmz: %0.5f"%(r[zmask & rmask & classA] - z[zmask & rmask & classA]).mean())
    print("Class A mean imz: %0.5f"%(i[zmask & imask & classA] - z[zmask & imask & classA]).mean())
    print("Class A mean rmi: %0.5f"%(r[imask & rmask & classA] - i[imask & rmask & classA]).mean())
    print("Class B mean rmz: %0.5f"%(r[zmask & rmask & ~classA] - z[zmask & rmask & ~classA]).mean())
    print("Class B mean imz: %0.5f"%(i[zmask & imask & ~classA] - z[zmask & imask & ~classA]).mean())
    print("Class B mean rmi: %0.5f"%(r[imask & rmask & ~classA] - i[imask & rmask & ~classA]).mean())
    divider = make_axes_locatable(axes[2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    #cbaxes = fig.add_axes([0.905, 0.13, 0.01, 0.77])
    cb = plt.colorbar(h1, cax = cax, ticks=[1,1.5, 2,2.5, 3])  
    cb.set_label(label, fontsize=16)
    
    axes[0].set_xlabel("$i' - z'$", fontsize=16)
    axes[1].set_xlabel("$r' - i'$", fontsize=16)
    axes[2].set_xlabel("$r' - z'$", fontsize=16)
    
    axes[0].set_ylabel("$M_{z'}$", fontsize=16)
    
    caa = mlines.Line2D([], [], color='#69CF37', markeredgecolor='#69CF37', linewidth=0, marker='o', linestyle='none', markersize=8, label='Class A')
    cbb = mlines.Line2D([], [], color='#1B737B', markeredgecolor='#1B737B', linewidth=0, marker='>', linestyle='none', markersize=8, label='Class B')
    axes[1].legend(handler_map={caa: HandlerLine2D(numpoints=1), cbb: HandlerLine2D(numpoints=1)}, handles=[caa, cbb]) #)

    #axes[1].set_ylabel("$i'$", fontsize=16)
    #axes[2].set_ylabel("$z'$", fontsize=16)

    plt.tight_layout()
    axes[0].locator_params(nbins=6)
    axes[1].locator_params(nbins=6)
    axes[2].locator_params(nbins=6)
    
    axes[0].axis('tight')
    axes[1].axis('tight')
    axes[2].axis('tight')

    #cax.locator_params(nbins=6)
    fig.savefig("colour.pdf", bbox_inches="tight")
    axes[1].legend(handler_map={caa: HandlerLine2D(numpoints=1), cbb: HandlerLine2D(numpoints=1)}, frameon=False, handles=[caa, cbb], bbox_to_anchor=(-0.2, 1.02, 1.4, .102), loc=3, ncol=2, mode="expand", borderaxespad=0.) #)

    fig.set_size_inches(6.5,2.5)
    axes[0].locator_params(nbins=4)
    axes[1].locator_params(nbins=4)
    axes[2].locator_params(nbins=4)
    axes[0].axvline(vimz, color="#888888", ls=":")
    axes[1].axvline(vrmi, color="#888888", ls=":")
    axes[2].axvline(vrmz, color="#888888", ls=":")
    #cax.locator_params(nbins=4)
    fig.savefig("colour.png", transparent=True, bbox_inches="tight", dpi=300)

    
def plotColourDiagrams2(cat, colourColumn='Chi2DeltaKingDiv', label=r"$\chi^2_{\rm{Delta}} / \chi^2_{\rm{King30}}$"):
    z = cat['Z_8']
    i = cat['I_8']
    r = cat['R_8']
    
    zmask = cat['Z_MASK'] & (cat['Z_8'] < 90)
    imask = cat['Z_MASK'] & (cat['I_8'] < 90)
    rmask = cat['Z_MASK'] & (cat['R_8'] < 90)
    
    
    mm = zmask & imask & rmask
    
    fig, axes = plt.subplots(figsize=(11,5), ncols=2, sharey=True)
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
    vmax = 3    
    vmin = 1
    
    h1 = axes[0].scatter(i[mm] - z[mm], r[mm] - z[mm], c=cat[mm][colourColumn], vmin=vmin, vmax=vmax, edgecolor="none", cmap=cmap)
    h2 = axes[1].scatter(r[mm] - i[mm], r[mm] - z[mm], c=cat[mm][colourColumn], vmin=vmin, vmax=vmax, edgecolor="none", cmap=cmap)
    
    divider = make_axes_locatable(axes[1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    #cbaxes = fig.add_axes([0.905, 0.13, 0.01, 0.77])
    cb = plt.colorbar(h1, cax = cax, ticks=[1, 2, 3])  
    cb.set_label(label, fontsize=16)
    
    axes[0].set_xlabel("$i' - z'$", fontsize=16)
    axes[1].set_xlabel("$r' - i'$", fontsize=16)
    
    axes[0].set_ylabel("$r' - z'$", fontsize=16)
    #axes[1].set_ylabel("$i'$", fontsize=16)
    #axes[2].set_ylabel("$z'$", fontsize=16)

    plt.tight_layout()
    fig.savefig("colour2.pdf", bbox_inches="tight")
    
def f(m, maxv, alpha, m50):
        return maxv * (1 - ((alpha * (m - m50))/(np.sqrt(1 + alpha * alpha * (m - m50) * (m - m50)))))
        
def fitCompleteness(classifier):
    
    

    xdata = classifier.sc.mags
    ydata = np.mean(classifier.sc.hRatio, axis=0)
    
    popt, pcov = curve_fit(f, xdata, ydata, p0=[0.4,1.5,15])
    
    fig, ax0 = plt.subplots(figsize=(5,4))
    ax0.set_xlabel("$z'$", fontsize=16)
    ax0.set_ylabel(r"$\rm{Completeness}$", fontsize=16)
    
    xdiff = np.diff(xdata)
    b1 = xdata[:-1] - xdiff*0.5
    b2 = np.array([xdata[-1]+xdiff[-1]*0.5])
    
    bine = np.concatenate((b1,b2))
    print(bine)
    
    ax0.plot(xdata, ydata, 'b-', linewidth=2, label="Classifier 2")
    #ax0.hist(xdata, bins=bine, weights=ydata, linewidth=2, histtype='step', label="Classifier 2")
    
    #x50 = interp1d(ydata,xdata)(0.5)
    #print(x50)
    
    yfit = f(xdata, *popt)
    ax0.plot(xdata, yfit, 'r--', linewidth=2,label="Fit")
    
    ax0.legend(loc=3)
    
    fig.savefig("../doc/images/completeness.pdf", bbox_inches='tight')
    
    return popt, pcov


def getSigmaZ(mz,mu):
    return 1.07 - 0.1 * (mz - mu + 22)
    
def getM0(mz, mu):
    return (-7.66 + 0.04 * (mz - mu)) + mu
    
def getMagnitudeInfo(zabs, maxv, alpha, m50):

    fig,ax0 = plt.subplots(figsize=(5,4))
    ax0.set_xlabel("$z'$", fontsize=16)
    ax0.set_ylabel("$N$", fontsize=16)
    
    bins = 20
    hist, be = np.histogram(zabs, bins=bins)
    bc = 0.5 * (be[:-1] + be[1:])
    xdata = bc
    adj = f(xdata, maxv, alpha, m50)
    good = adj > 0.3
    print(adj)
    
    hist2 = (hist / adj)
    hist3 = hist2[good]
    xdata2 = xdata[good]
    mz = getMZ()
    mu = 27.39
    def ff(m, a0):
        sigmam = getSigmaZ(mz, mu)
        m0 = getM0(mz, mu)
        return (a0 / ((np.sqrt(2*np.pi)) * sigmam)) * np.exp(-(m - m0)*(m - m0)/(2*sigmam*sigmam))
    
    popt, pcov = curve_fit(ff, xdata2, hist3, p0=[2500])
    print(popt)
    print(pcov)
    a0 = popt[0]
    x = np.linspace(xdata.min(), xdata.max(), 100)
    yfit = ff(x, *popt)

    #yfunct = (a0 / ((np.sqrt(2*np.pi)) * sigmam)) * np.exp(-(x - m0)*(x - m0)/(2*sigmam*sigmam) )
    #print(xdata, yfit)
    
    ax0.hist(bc, bins=be, weights=hist, histtype='step', linewidth=1, label="Observed distribution")
    #ax0.hist(xdata, bins=be, weights=hist2, histtype='step', linewidth=1, label="Corrected distribution")
    ax0.hist(xdata2, bins=be, weights=hist3, histtype='step', linewidth=1, label="Corrected distribution")
    ax0.plot(x, yfit, 'r--', label="Observed model")
    #ax0.plot(x, yfunct, 'r', label="Underlying distribution")
    #ax0.plot(xdata, 70*(1 - (alpha * (xdata - m50)) / np.sqrt(1 + alpha * alpha * (xdata - m50) * (xdata - m50))), ls="--")
    ax0.legend(loc=2)

def updateRaAndDec(g, cat):
    t = 1.5 / (3600.)
    g = g.copy()
    for i,row in enumerate(g):
        ra = row['RA']
        dec = row['DEC']
        
        dists = np.sqrt( (cat['X_WORLD']-ra)*(cat['X_WORLD']-ra) + (cat['Y_WORLD']-dec)*(cat['Y_WORLD']-dec))
        minDist = dists.argmin()
        if (dists[minDist] < t):
            row['RA'] = cat['X_WORLD'][minDist]
            row['DEC'] = cat['Y_WORLD'][minDist]
        else:
            print(dists[minDist]*3600)
    return g
    
def getOptimalBinSize(x, x_min=None, x_max=None):

    if x_max is None:
        x_max = max(x)
    if x_min is None:
        x_min = min(x)
    N_MIN = 4   #Minimum number of bins (integer)
                #N_MIN must be more than 1 (N_MIN > 1).
    N_MAX = 50  #Maximum number of bins (integer)
    N = range(N_MIN,N_MAX) # #of Bins
    N = np.array(N)
    D = (x_max-x_min)/N    #Bin size vector
    C = np.zeros(shape=(np.size(D),1))
    
    #Computation of the cost function
    for i in xrange(np.size(N)):
        edges = np.linspace(x_min,x_max,N[i]+1) # Bin edges
        ki = np.histogram(x,edges) # Count # of events in bins
        ki = ki[0]    
        k = np.mean(ki) #Mean of event count
        v = np.sum((ki-k)**2)/N[i] #Variance of event count
        C[i] = (2*k-v)/((D[i])**2) #The cost Function
    #Optimal Bin Size Selection
    
    cmin = np.min(C)
    idx  = np.where(C==cmin)
    idx = int(idx[0])
    optD = D[idx]
    
    edges = np.linspace(x_min,x_max,N[idx]+1)
    return edges