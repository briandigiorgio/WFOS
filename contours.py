#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
from optexttools import *

if __name__ == '__main__':
    fwhm = .9
    npix = 1000
    arcsec = 100 #size of an arcsecond in pixels
    plotsize = 2 #arcseconds away from 0 to show
    lim = npix//(2*arcsec) #upper and lower bounds of the psf
    x = np.linspace(-lim,lim,npix)
    y = np.linspace(-lim,lim,npix)

    #get psf with peak = 1, normalize it to find fraction of flux in each
    #pixel, multiply to get how much flux that would be in a whole arcsec
    tpsf = moffat((0,0), x, y, fwhm)
    psf = (tpsf/np.sum(tpsf)) * arcsec**2 

    #values of the psf at certain magnitudes if central mag is 24
    fracs = 100**(-.2*np.arange(5,-4,-1))
    labels = dict(zip(fracs,np.arange(29,20,-1).astype(str)))

    plt.imshow(psf, aspect = 'equal', cmap = 'inferno')
    #plt.colorbar()
    contours = plt.contour(psf, levels = fracs, colors = 'white')
    plt.xticks(np.arange(0,npix+1,npix//(4*lim)), np.arange(-lim,lim+.01,.5))
    plt.yticks(np.arange(0,npix+1,npix//(4*lim)), np.arange(-lim,lim+.01,.5))
    plt.xlabel('Arcseconds')
    plt.title(r'Moffat PSF, $\beta = 2.9$, FWHM = %g", Total Flux = 24 Mag' %
            fwhm)
    ax = plt.gca()
    ax.set_xlim(int(npix/2-plotsize*arcsec), int(npix/2+plotsize*arcsec))
    ax.set_ylim(int(npix/2-plotsize*arcsec), int(npix/2+plotsize*arcsec))
    plt.clabel(contours, inline = 1, use_clabeltext = True, fmt = labels)

    plt.tight_layout()
    plt.show()

