#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from optexttools import *
import time

if __name__ == '__main__':
    start = time.time()
    fwhms = np.linspace(.3, .9, 7)
    npix = 1000
    arcsec = 100
    #slitlength = 1
    fibersize = .33
    t = 1.5/2

    #slit75 = makerect(int(slitlength*arcsec), int(.75 * arcsec))
    #slit25 = makerect(int(slitlength*arcsec), int(.25 * arcsec))
    #hex7 = makebundle(7, fibersize, npix = npix, arcsec = arcsec)
    #hex19 = makebundle(19, fibersize, npix = npix, arcsec = arcsec)

    slit75f = np.zeros(len(fwhms))
    slit25f = np.zeros(len(fwhms))
    hex7f = np.zeros(len(fwhms))
    hex19f = np.zeros(len(fwhms))

    counter = 0
    total = len(fwhms)
    progtime = time.time()
    progressbar(counter, total)

    for i in range(len(fwhms)):
        slitlength = 1.5*fwhms[i]
        slit75 = makerect(int(slitlength*arcsec), int(.75 * arcsec))
        slit25 = makerect(int(slitlength*arcsec), int(.25 * arcsec))
        hex7 = makebundle(7, fibersize, npix = npix, arcsec = arcsec)
        hex19 = makebundle(19, fibersize, npix = npix, arcsec = arcsec)

        slit75f[i] = lossfrac(slit75, fwhms[i], npix = npix, trunc = t*fwhms[i])
        slit25f[i] = lossfrac(slit25, fwhms[i], npix = npix, trunc = t*fwhms[i])
        hex7f[i] = lossfrac(hex7, fwhms[i], npix = npix, trunc = t*fwhms[i])
        hex19f[i] = lossfrac(hex19, fwhms[i], npix = npix, trunc = t*fwhms[i])
        counter+= 1
        progtime = progressbar(counter, total, progtime)

    plt.figure(figsize = (10,4))
    plt.subplot(121)
    plt.plot(fwhms, slit75f, label = '.75" slit')
    plt.plot(fwhms, slit25f, label = '.25" slit')
    plt.plot(fwhms, hex7f, '--', label = '7 fiber bundle')
    plt.plot(fwhms, hex19f, '--', label = '19 fiber bundle')
    plt.xlabel('FWHM')
    plt.ylabel('Fraction of flux captured')
    plt.legend()
    plt.grid(True)

    plt.subplot(122)
    plt.plot(fwhms, slit75f/slit75f, label = '.75" slit')
    plt.plot(fwhms, slit25f/slit75f, label = '.25" slit')
    plt.plot(fwhms, hex7f/slit75f, '--', label = '7 fiber bundle')
    plt.plot(fwhms, hex19f/slit75f, '--', label = '19 fiber bundle')
    plt.xlabel('FWHM')
    plt.ylabel('Fraction of flux captured\n(normalized to 75" slit flux)')
    plt.legend()
    plt.grid(True)

    print(time.time() - start)
    plt.tight_layout()
    plt.show()

