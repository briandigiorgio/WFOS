#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
from optexttools import *

if __name__ == "__main__":
    fibers = [1,2,4,8,16,32]
    nruns = len(fibers)
    ff = 10000

    flux = np.zeros(nruns)
    yerr = np.zeros(nruns)
    sn = np.zeros(nruns)

    for i in range(nruns):
        noise = 10000/(fibers[i]**2)
        flux[i], yerr[i], sn[i] = optext(fibers[i], fwhm = .5,
                                         ff = ff, noise = noise, a = True)
        print(flux[i], yerr[i], sn[i])

    plt.plot(fibers, sn, c = 'orange', label = "S/N")
    plt.plot(fibers, sn, 'o', c = 'orange')

    #plt.plot(fibers, flux, '-', label = 'Flux')
    #plt.fill_between(fibers, flux-yerr, flux+yerr, facecolor='b', alpha=.2)
    #plt.axhline(y=ff, ls = '--', c= 'r', label = 'Theoretical Flux')

    plt.title("Optimally Extracted Values for Different Numbers of Fibers")
    plt.xlabel("Number of Fibers on a Side")
    plt.grid(True)
    plt.legend()
    #plt.savefig("snplot.png")
    plt.show()
