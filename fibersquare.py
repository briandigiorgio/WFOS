#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as stats
from scipy.signal import convolve2d

def gaussian(x, mean, std):
    return (1/(std*np.sqrt(2*np.pi)))**np.exp(-1*((x-mean)**2)/(2*std**2))

def addnoise(psf, noise):
    return psf + np.random.normal(0, noise, size = psf.shape)

def optimalextract(data, tdata, var):
    #method from Naylor 1998 Eqs. 9-12
    #data = observed fluxes in fibers, tdata = theoretical psf values
    #returns flux, error, S/N ratio
    sumvar = np.sum(np.square(tdata)/var)
    weights = (tdata/var)/sumvar
    flux = np.sum(weights * data)
    error = np.sqrt(np.sum(np.square(weights) * var))
    sn = flux * np.sqrt(sumvar)
    return flux, error, sn 

if __name__ == "__main__":
    #define parameters
    fwhm = .5 #seeing fwhm
    npix = 100 #number of points in the theoretical psf to make
    nfibers = 10 #number of fibers on each side (total num is nfibers^2)
    fluxfactor = 10000 #total amount of flux in final psf
    noise = 50 #stdev of noise added to theoretical psf
    noise2 = .005

    #define variables
    std = fwhm/2.355 #convert fwhm to stdev
    fibersize = npix//nfibers #size of fiber in pixels
    #var = np.square(noise) + np.square(noise2) #variance of noise
    fiber = np.ones((fibersize,fibersize)) #square mask to represent pixel

    #make grid of positions across pixel space
    x = np.linspace(-1,1,npix)
    y = np.linspace(-1,1,npix)
    pos = [[[x[i],y[j]] for j in range(npix)] for i in range(npix)]

    #define 2d gaussian pdf, evaluate to make a grid of values, normalize
    pdf = stats.multivariate_normal([0,0], [[std,0],[0,std]])
    tpsf = pdf.pdf(pos)
    tpsf = tpsf/np.sum(tpsf)

    #add noise so it's not perfect
    #psf = fluxfactor * addnoise(tpsf, noise)
    psf = tpsf * fluxfactor

    #plt.imshow(psf, cmap = "inferno")
    #plt.colorbar()
    #plt.show()

    #convolve fibers with psfs to get simulated fiber data
    alldata = convolve2d(psf, fiber, mode = 'valid')
    talldata = convolve2d(tpsf, fiber, mode = 'valid')

    #select only the data that would correspond to unoverlapping fibers
    #fibersize = (alldata.shape[0] - 1)/nfibers + 1
    #print fibersize
    xpos = list(range(0,alldata.shape[0]-fibersize,fibersize))\
            +[alldata.shape[0]-1]
    ypos = list(range(0,alldata.shape[1]-fibersize,fibersize))\
            +[alldata.shape[1]-1]
    data = alldata[xpos,:][:,ypos]
    tdata = talldata[xpos,:][:,ypos]
    #data = addnoise(data, np.sqrt(noise**2 + data))
    data = np.random.poisson(data)
    data = addnoise(data, noise)
    data = data
    var =  noise**2 + data

    plt.imshow(data, cmap = 'inferno')
    plt.colorbar()
    plt.show()

    flux, variance, sn = optimalextract(data, tdata, var)

    print("Sum of Original PSF: %f" % (np.sum(psf)))
    print("Flux by Optimal Extraction: %f, Error: %f"%(flux,variance))
    print("S/N: %f" % sn)
