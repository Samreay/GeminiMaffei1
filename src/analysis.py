import os
import sys
import numpy as np
import matplotlib, matplotlib.pyplot as plt
import sextractor

# Create a SExtractor instance
sex = sextractor.SExtractor()

# Modify the SExtractor configuration
sex.config['GAIN'] = 0.938
sex.config['PIXEL_SCALE'] = .19
sex.config['VERBOSE_TYPE'] = "FULL"
sex.config['CHECKIMAGE_TYPE'] = "BACKGROUND"

# Add a parameter to the parameter list
sex.config['PARAMETERS_LIST'].append('FLUX_BEST')

# Lauch SExtractor on a FITS file
#sex.run("nf260002.fits")

# Read the resulting catalog [first method, whole catalog at once]
#catalog = sex.catalog()
plt.imshow(np.random.random(size=(10,10)))