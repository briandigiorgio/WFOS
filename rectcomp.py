#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
from optexttools import *
import time

if __name__ == "__main__":
    start = time.time()
    npix = 180
    #sizes = np.asarray((6,9,18,24,36,45))/npix
    sizes = np.asarray((9,24,45,90,180))/npix
    pixels = 1/sizes
    print(sizes,pixels)
    ff = 10000
    noises = [1,100,1000,10000]
    shape = (len(noises), len(sizes))
    cmap = make_cmap(len(noises), 'inferno')
    nruns = 50
    normed = False
    fwhm = .3

    flux = np.zeros(shape)
    yerr = np.zeros(shape)
    sn = np.zeros(shape)
    aflux = np.zeros(shape)
    ayerr = np.zeros(shape)
    asn = np.zeros(shape)
    snnorm = np.zeros(shape)
    asnnorm = np.zeros(shape)
    sns = np.zeros(nruns)
    asns = np.zeros(nruns)
    errors = np.zeros(nruns)
    aerrors = np.zeros(nruns)

    counter = 0
    total = len(noises)*len(sizes)
    progtime = time.time()

    for i in range(len(noises)):
        for j in range(len(sizes)):
            noise = noises[i]/np.sqrt(pixels[j])
            for k in range(nruns):
                flux[i][j], errors[k], sns[k] = optrext(sizes[j], fwhm = fwhm,
                                        npix = npix, ff = ff, noise = noise)
                aflux[i][j], aerrors[k], asns[k] = optrext(sizes[j], fwhm=fwhm,
                            npix = npix, ff = ff, noise = noise, a = True)

            sn[i][j] = np.average(sns)
            asn[i][j] = np.average(asns)
            #snnorm[i][j] = sn[i][j]*np.sqrt(ff+noises[i]**2)/ff
            #asnnorm[i][j] = asn[i][j]*np.sqrt(ff+noises[i]**2)/ff
            yerr[i][j] = np.sqrt(np.sum(np.power(errors/nruns, 2)))
            ayerr[i][j] = np.sqrt(np.sum(np.power(aerrors/nruns, 2)))

            counter += 1
            progtime = progressbar(counter, total, progtime)
            
        if normed:
            plt.loglog(sizes, snnorm[i], c = cmap[i], label = noises[i])
            plt.loglog(sizes, asnnorm[i], c = cmap[i], ls = '--')
        else:
            plt.semilogy(sizes, sn[i], c = cmap[i], label =
                    (noises[i]**2)/ff)
            plt.semilogy(sizes, asn[i], c = cmap[i], ls = '--')
            #plt.fill_between(fibers, sn[i]-yerr[i], sn[i]+yerr[i], facecolor =
            #        cmap[i], alpha = .2)

    plt.title("S/N Ratios for Different Background Noise with FWHM = " +
            str(fwhm) + '"')
    plt.xlabel("Size of Pixels Along Slit in arcseconds")
    plt.ylabel("S/N")
    plt.grid(True)
    plt.legend(title = "Background/Source")
    print("\nTime taken: ", time.time()-start)
    #print(yerr)
    plt.show()
