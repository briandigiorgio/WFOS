#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np
from optexttools import *

if __name__ == "__main__":
    fibers = 4
    nruns = 1000
    ff = 10000
    noise = 10000/(fibers**2)

    flux = np.zeros(nruns)
    yerr = np.zeros(nruns)
    sn = np.zeros(nruns)

    aflux = np.zeros(nruns)
    ayerr = np.zeros(nruns)
    asn = np.zeros(nruns)

    print(difftest(fibers, fwhm = .5, ff = ff, noise = noise))

    for i in range(nruns):
        flux[i], yerr[i], sn[i] = optext(fibers, fwhm = .5,
                                        ff = ff, noise = noise)

        aflux[i], ayerr[i], asn[i] = optext(fibers, fwhm = .5,
                                             ff = ff, noise = noise, a = True)
    avgsn = np.average(sn)
    avgasn = np.average(asn)
    print('Fluxes: ',np.average(flux), np.average(aflux))
    print('S/Ns: ', avgsn, avgasn)
    print("Improvement: %f percent" % diff(avgsn, avgasn))

    plt.hist(flux) 
    tsd = np.average(yerr)
    mean = np.average(flux)
    sd = np.std(flux)

    plt.axvline(x=ff, ls = '-', c= 'r', label = 'Theoretical Flux')
    plt.axvline(x = ff-tsd, ls = '--', c = 'r')
    plt.axvline(x = ff+tsd, ls = '--', c = 'r')

    plt.axvline(x = mean, ls = "-", c = "k", label = 'Average Flux')
    plt.axvline(x = mean+sd, ls = "--", c = "k")
    plt.axvline(x = mean-sd, ls = "--", c = "k")

    print('Diff from theory: ', mean-ff, sd-tsd)

    plt.title("Optimally Extracted Fluxes with %d Samples" % nruns)
    plt.xlabel("Flux Values")
    plt.grid(True)
    plt.legend()
    plt.show()
