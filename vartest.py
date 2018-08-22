#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
from optexttools import *
import time

if __name__ == "__main__":
    start = time.time()
    fibers = [1,2,4,8,16,32]
    ff = 10000
    #ff = [100,1000,10000,100000]
    noises = [1,100,1000,10000,100000]
    #noises = 1000
    shape = (len(noises), len(fibers))
    #shape = (len(ff), len(fibers))
    cmap = make_cmap(len(noises), 'inferno')
    #cmap = make_cmap(len(ff), 'inferno')
    nruns = 100
    normed = False
    fignum = (len(noises)*100) + (len(fibers)*10) + 1

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
    total = len(noises)*len(fibers)
    progtime = time.time()

    for i in range(len(noises)):
    #for i in range(len(ff)):
        plt.figure(figsize=(18,5))
        for j in range(len(fibers)):
            var = np.zeros((fibers[j],fibers[j]))
            vara = np.zeros_like(var)
            noise = noises[i]/(fibers[j]**2)
            #noise = noises/(fibers[j]**2)
            for k in range(nruns):
                var += optext(fibers[j], fwhm = .8,
                                                 ff = ff, noise = noise)
                vara += optext(fibers[j], fwhm = .8,
                            ff = ff, noise = noise, a = True)

            counter += 1
            progtime = progressbar(counter, total, progtime)
            
            plt.subplot(fignum+j)
            var = var/nruns
            vara = vara/nruns
            plt.imshow(vara, cmap = 'inferno')
            plt.colorbar()
            print(noises[i], fibers[j], np.sum(vara))

    plt.tight_layout()
    print("\nTime taken: ", time.time()-start)
    plt.show()
