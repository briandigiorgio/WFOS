#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
import argparse
from optexttools import *

if __name__ == "__main__":
    fibers = 64 
    nruns = len(fibers)
    ff = 1
    
    flux = np.zeros(nruns)
    var = np.zeros(nruns)
    sn = np.zeros(nruns)

    for i in range(nruns):
        flux[i], var[i], sn[i] = optext(fibers[i], fwhm = .5, 
                noise = .00005, ff = ff)

    yerr = np.sqrt(var)
    #plt.plot(fibers, flux, label = "Flux")
    #plt.plot(fibers, var, label = "Variance")
    #plt.errorbar(fibers, flux, yerr = np.sqrt(var), fmt = '-', label = "Flux")
    plt.plot(fibers, flux, '-', label = 'Flux')
    plt.fill_between(fibers, flux-yerr, flux+yerr, facecolor='b', alpha=.2)
    plt.plot(fibers, sn, label = "S/N")
    plt.plot(fibers,sn,'o', c = 'orange')
    plt.axhline(y=ff, ls = '--', c= 'r', label = 'Theoretical Flux')

    plt.title("Optimally Extracted Values for Different Numbers of Fibers")
    plt.xlabel("Number of Fibers on a Side")
    plt.grid(True)
    plt.legend()
    plt.savefig("snplot.png")
    plt.show()
