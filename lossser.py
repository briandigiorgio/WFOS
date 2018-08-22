#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from optexttools import *
import time

if __name__ == '__main__':
    start = time.time()
    npix = 1000
    arcsec = 100
    slitlength = 1
    fibersize = .33
    t = 0
    n = [.5,1,1.5,2,4,6,10]

    slit75 = makerect(int(slitlength*arcsec), int(.75 * arcsec))
    slit25 = makerect(int(slitlength*arcsec), int(.25 * arcsec))
    hex7 = makebundle(7, fibersize, npix = npix, arcsec = arcsec)
    hex19 = makebundle(19, fibersize, npix = npix, arcsec = arcsec)

    slit75f = np.zeros(len(n))
    slit25f = np.zeros(len(n))
    hex7f = np.zeros(len(n))
    hex19f = np.zeros(len(n))

    counter = 0
    total = len(n)
    progtime = time.time()
    progressbar(counter, total)

    for i in range(len(n)):
        slit75f[i] = lossfrac(slit75, n[i], npix = npix, psf = 's')
        slit25f[i] = lossfrac(slit25, n[i], npix = npix, psf = 's')
        hex7f[i] = lossfrac(hex7, n[i], npix = npix, trunc = t,
                psf = 's')
        hex19f[i] = lossfrac(hex19, n[i], npix = npix, trunc = t,
                psf = 's')
        counter+= 1
        progtime = progressbar(counter, total, progtime)

    plt.figure(figsize = (10,4))
    plt.subplot(121)
    plt.plot(n, slit75f, label = '.75" slit')
    plt.plot(n, slit25f, label = '.25" slit')
    plt.plot(n, hex7f, '--', label = '7 fiber bundle')
    plt.plot(n, hex19f, '--', label = '19 fiber bundle')
    plt.xlabel('Sersic Index n')
    plt.ylabel('Fraction of flux captured')
    plt.legend()
    plt.grid(True)

    plt.subplot(122)
    plt.plot(n, slit75f/slit75f, label = '.75" slit')
    plt.plot(n, slit25f/slit75f, label = '.25" slit')
    plt.plot(n, hex7f/slit75f, '--', label = '7 fiber bundle')
    plt.plot(n, hex19f/slit75f, '--', label = '19 fiber bundle')
    plt.xlabel('Sersic Index n ')
    plt.ylabel('Fraction of flux captured\n(normalized to 75" slit flux)')
    plt.legend()
    plt.grid(True)

    print(time.time() - start)
    plt.tight_layout()
    plt.show()

