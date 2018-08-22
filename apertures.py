#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from optexttools import *

if __name__ == '__main__':
    npix = 2000
    arcsec = 1000
    slitlength = .45
    fibersize = .33
    center = (npix//2, npix//2)

    background = np.zeros((npix,npix))
    slit25 = makerect(int(slitlength*arcsec), int(.25*arcsec))
    slit75 = makerect(int(slitlength*arcsec), int(.75*arcsec))
    bundle7 = makebundle(7, fibersize, npix = npix, arcsec = arcsec)
    bundle19 = makebundle(19, fibersize, npix = npix, arcsec = arcsec)
    trunc = makecircle(slitlength * arcsec)

    #background = addfiber(slit25, background, center)
    background = addfiber(slit75, background, center)
    background = addfiber(bundle7, background, center)
    #background = addfiber(bundle19, background, center)
    background = addfiber(trunc, background, center)

    plt.imshow(background, cmap = 'inferno')
    ax = plt.gca()
    ax.tick_params(bottom = 0, left = 0, labelbottom = 0, labelleft = 0)
    plt.tight_layout()
    plt.show()

