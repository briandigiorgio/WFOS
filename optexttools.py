#!/usr/bin/env python

import time
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as stats
from scipy.signal import convolve2d
from matplotlib import colors, rc, cm
from astropy.modeling import functional_models as dists
from skimage.transform import resize
from scipy.optimize import least_squares

def gaussian(x, mean, std, norm = False):
    psf = (1/(std*np.sqrt(2*np.pi)))**np.exp(-1*((x-mean)**2)/(2*std**2))
    if norm:
        return psf/psf.sum()
    else:
        return psf

def addnoise(psf, noise):
    return psf + np.random.normal(0, noise, size = psf.shape)

#returns a moffat psf centered at [x,y] = center evaluated at pos (2darray)
#beta = 2.9 is given by Gemini GLAO in 
#http://www.ctio.noao.edu/~atokovin/papers/Gemini-GLAO.pdf
#parameter naming is messed up because astropy uses unconventional names
def moffat(center, x, y, fwhm, beta = 2.9, norm = True):
    alpha = fwhm/(2*np.sqrt(2**(1/beta)-1))
    tpsf = np.zeros((len(x),len(y)))
    for i, xi in enumerate(x):
        for j, yj in enumerate(y):
            tpsf[i,j] = dists.Moffat2D.evaluate(xi, yj, 1, x_0 = center[0],
                    y_0 = center[1], gamma = alpha, alpha = beta)
    if norm:
        return tpsf/tpsf.sum()
    else:
        return tpsf


def optimalextract(data, tdata, var, aperture = False):
    #method from Naylor 1998 Eqs. 9-12
    #data = observed counts in fibers, tdata = theoretical psf values
    #returns flux, error on flux measurement, S/N ratio
    sumvar = np.sum(np.square(tdata)/var)
    weights = (tdata/var)/sumvar

    if aperture:
        weights = np.ones((data.shape))

    flux = np.sum(weights * data)
    error = np.sqrt(np.sum(np.square(weights) * var))
    #sn = flux * np.sqrt(sumvar)
    sn = flux / error
    return flux, error, sn 

def apsum(data, var):
    flux = np.sum(data) 
    error = np.sqrt(np.sum(var))
    sn = flux/error
    return flux, error, sn

def optext(nfibers, fwhm, npix = 128, noise = 50, ff = 10000, a = False, m =
        True):
    #define variables
    std = fwhm/2.355 #convert fwhm to stdev
    fibersize = npix//nfibers #size of fiber in pixels
    fiber = np.ones((fibersize,fibersize)) #square mask to represent pixel
    size = 1

    #make grid of positions across pixel space
    x = np.linspace(-.5*size,.5*size,npix)
    y = np.linspace(-.5*size,.5*size,npix)
    pos = [[[x[i],y[j]] for j in range(npix)] for i in range(npix)]

    #define 2d pdf, evaluate to make a grid of values, normalize
    if m:
        tpsf = moffat((0,0), x, y, fwhm, 2.9)
        tpsf = tpsf/np.sum(tpsf)
    else:
        pdf = stats.multivariate_normal([0,0], [[std,0],[0,std]])
        tpsf = pdf.pdf(pos)

    tpsf = tpsf/np.sum(tpsf)
    psf = tpsf * ff

    #convolve fibers with psfs to get simulated fiber data
    alldata = convolve2d(psf, fiber, mode = 'valid')
    talldata = convolve2d(tpsf, fiber, mode = 'valid')

    #select only the data that would correspond to unoverlapping fibers
    xpos = list(range(0,alldata.shape[0]-fibersize,fibersize))\
        +[alldata.shape[0]-1]
    ypos = list(range(0,alldata.shape[1]-fibersize,fibersize))\
        +[alldata.shape[1]-1]
    data = np.random.poisson(alldata[xpos,:][:,ypos])
    tdata = talldata[xpos,:][:,ypos]
    data = addnoise(data, noise)
    var = np.square(noise) + data #variance of noise
    '''
    plt.imshow(data, cmap = "inferno")
    plt.title("Noise = %d" % noise)
    plt.show()
    '''
    #if a:
    #    return apsum(data, var)
    return optimalextract(data, tdata, var, aperture = a) 
    #return var


def difftest(nfibers, fwhm, npix = 128, noise = 50, ff = 10000, a = False):
    #define variables
    std = fwhm/2.355 #convert fwhm to stdev
    fibersize = npix//nfibers #size of fiber in pixels
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
    #psf = ff * addnoise(tpsf, noise)
    psf = tpsf * ff

    #convolve fibers with psfs to get simulated fiber data
    alldata = convolve2d(psf, fiber, mode = 'valid')
    talldata = convolve2d(tpsf, fiber, mode = 'valid')

    #select only the data that would correspond to unoverlapping fibers
    xpos = list(range(0,alldata.shape[0]-fibersize,fibersize))\
        +[alldata.shape[0]-1]
    ypos = list(range(0,alldata.shape[1]-fibersize,fibersize))\
        +[alldata.shape[1]-1]
    data = np.random.poisson(alldata[xpos,:][:,ypos])
    tdata = talldata[xpos,:][:,ypos]
    data = addnoise(data, noise)
    var = np.square(noise) + data #variance of noise

    #plt.imshow(data, cmap = "inferno")
    #plt.colorbar()
    #plt.show()

    flux, error, sn = optimalextract(data, tdata, var, aperture = False)
    aflux, aerror, asn = optimalextract(data, tdata, var, aperture = True)

    return diff(flux, aflux), diff(error, aerror), diff(sn, asn)

def diff(orig, new):
    return ((new - orig)/orig) * 100

#makes an array of length numlines going through the specified color map
#plt.plot(..., c = colors[i]) when plotting multiple lines
def make_cmap(numlines, cmap):
    cnorm = colors.Normalize(vmin = 0, vmax = numlines)
    scalarmap = cm.ScalarMappable(norm = cnorm, cmap = cmap)
    return scalarmap.to_rgba(range(numlines))

def progressbar(progress, total, progtime = 0, tlast = 0):
    end = ''
    print(' '*73, end = '\r')
    if not progtime:
        end = '\r'
    print('Progress: [%s%s] %d%%' % ('▒' * progress, '_' * (total-progress), 
         (progress * 100)//total), end = end)
    if progtime:
        timediff = float(time.time()-progtime)

        if tlast:
            new = np.array(len(tlast))
            for i in range(len(tlast)):
                new[i] = tlast[i]/(total-progress + i)
            new = new.append(timediff)
            print("  Time Remaining: ~%d seconds" %
                   ((total-progress)*np.average(new)), end = '\r')
            return time.time(), new

        print("  Time Remaining: ~%d seconds" %
                ((total-progress)*timediff), end = '\r')
            
        return time.time()

#returns a hexagonal mask array with desired radius
#radius is from corner to corner (longer dimension of hexagon)
def makehex(rad):
    r3 = np.sqrt(3)
    shape = (2*rad,int(r3*rad))
    hexagon = np.ones(shape)

    #change pixels that are outside of hexagon
    for x in range(shape[0]):
        for y in range(shape[1]):
            if y > r3/2*rad + r3*x or y < r3/2*rad - r3*x \
            or y > -r3*x + 5*r3/2*rad or y < r3*x - 3*r3/2*rad:
                hexagon[x,y] = 0
    return hexagon

#convolve a fiber onto a psf at only one specified location
#input center of the fiber as (y,x) coordinate
def convolve(center, fiber, psf):
    odd = (fiber.shape[0]%2, fiber.shape[1]%2) #accounts for odd dimensions
    xoffset = fiber.shape[1]//2
    yoffset = fiber.shape[0]//2

    #multiplies arrays together
    return np.sum(fiber * psf[center[0]-yoffset:center[0]+yoffset+odd[0],
            center[1]-xoffset:center[1]+xoffset+odd[1]])

#performs optimal extraction simulation using a first order hexagonal IFU
#first order meaning 1 outer ring of fibers (7 fiber)
def opthext1(fwhm, npix = 128, size = 1, noise = 1000, ff = 10000, a = False, m
        = True, s = True):
    #define variables
    r3 = np.sqrt(3)
    std = fwhm/2.355 #convert fwhm to stdev
    r = int(npix/(3*r3))
    fiber = makehex(r) #square mask to represent pixel
    center = npix//2

    #make grid of positions across pixel space
    x = np.linspace(-.5*size,.5*size,npix)
    y = np.linspace(-.5*size,.5*size,npix)
    pos = [[[x[i],y[j]] for j in range(npix)] for i in range(npix)]

    if m:
        tpsf = moffat((0,0), x, y, fwhm, 2.9)
        tpsf = tpsf/np.sum(tpsf)
    else:
        pdf = stats.multivariate_normal([0,0], [[std,0],[0,std]])
        tpsf = pdf.pdf(pos)

    tpsf = tpsf/np.sum(tpsf)
    psf = tpsf * ff
    
    #manually goes through and finds values for all of the fiber locations
    #there's probably a more elegant way to do this
    #letters start at theta=0 and progress counterclockwise
    f0  = convolve((center,center), fiber, psf)
    f1a = convolve((center, int(center+r*r3)), fiber, psf)
    f1b = convolve((int(center+1.5*r), int(center+(r*r3)/2)), fiber, psf)
    f1c = convolve((int(center+1.5*r), int(center-(r*r3)/2)), fiber, psf)
    f1d = convolve((center, int(center-r*r3)), fiber, psf)
    f1e = convolve((int(center-1.5*r), int(center-(r*r3)/2)), fiber, psf)
    f1f = convolve((int(center-1.5*r), int(center+(r*r3)/2)), fiber, psf)

    #put all data into an array as it would be line by line
    #in case I ever figure out how to plot this
    data =   [f1c,f1b,
            f1d,f0,f1a,
              f1e,f1f]
    tdata = data/np.sum(data)

    #add noise, find variance, do optext
    data = addnoise(np.random.poisson(data), noise)
    var = np.square(noise) + data #variance of noise
    if s:
        return optimalextract(data, tdata, var, aperture = a) 
    else:
        return data, tdata, var

#same as opthext1 but for a 2nd order bundle (2 rings, 19 fibers)
def opthext2(fwhm, npix = 128, size = 1, noise = 1000, ff = 10000, a = False, m
        = True):
    #define variables
    r3 = np.sqrt(3)
    std = fwhm/2.355 #convert fwhm to stdev
    r = int(npix/(5*r3))
    fiber = makehex(r) #square mask to represent pixel
    center = npix//2

    #make grid of positions across pixel space
    x = np.linspace(-.5*size,.5*size,npix)
    y = np.linspace(-.5*size,.5*size,npix)
    pos = [[[x[i],y[j]] for j in range(npix)] for i in range(npix)]

    #define 2d pdf, evaluate to make a grid of values, normalize
    if m:
        tpsf = moffat((0,0), x, y, fwhm, 2.9)
        tpsf = tpsf/np.sum(tpsf)
    else:
        pdf = stats.multivariate_normal([0,0], [[std,0],[0,std]])
        tpsf = pdf.pdf(pos)

    tpsf = tpsf/np.sum(tpsf)
    psf = tpsf * ff
    
    #make all of the different fibers
    #there must be a better way to do this
    f0 = convolve((center,center), fiber, psf)

    f1a = convolve((center, int(center+r*r3)), fiber, psf)
    f1b = convolve((int(center+1.5*r), int(center+(r*r3)/2)), fiber, psf)
    f1c = convolve((int(center+1.5*r), int(center-(r*r3)/2)), fiber, psf)
    f1d = convolve((center, int(center-r*r3)), fiber, psf)
    f1e = convolve((int(center-1.5*r), int(center-(r*r3)/2)), fiber, psf)
    f1f = convolve((int(center-1.5*r), int(center+(r*r3)/2)), fiber, psf)

    f2a = convolve((center, int(center+2*r*r3)), fiber, psf)
    f2b = convolve((int(center+1.5*r), int(center+1.5*r*r3)), fiber, psf)
    f2c = convolve((center+3*r, int(center+r*r3)), fiber, psf)
    f2d = convolve((center+3*r, center), fiber, psf)
    f2e = convolve((center+3*r, int(center-r*r3)), fiber, psf)
    f2f = convolve((int(center+1.5*r), int(center-1.5*r*r3)), fiber, psf)
    f2g = convolve((center, int(center-2*r*r3)), fiber, psf)
    f2h = convolve((int(center-1.5*r), int(center-1.5*r*r3)), fiber, psf)
    f2i = convolve((center-3*r, int(center-r*r3)), fiber, psf)
    f2j = convolve((center-3*r, center), fiber, psf)
    f2k = convolve((center-3*r, int(center+r*r3)), fiber, psf)
    f2l = convolve((int(center-1.5*r), int(center+1.5*r*r3)), fiber, psf)

    #put all of the data together into an array scanning line by line
    #the array is a graphical representation of where all of the fibers are
    data =  [f2e,f2d,f2c,
           f2f,f1c,f1b,f2b,
         f2g,f1d,f0,f1a,f2a,
           f2h,f1e,f1f,f2l,
             f2i,f2j,f2k]
    tdata = data/np.sum(data)

    data = addnoise(np.random.poisson(data), noise)
    var = np.square(noise) + data #variance of noise
    return optimalextract(data, tdata, var, aperture = a)

#one hexagonal fiber, some stuff had to be rewritten so it would work
def opthext0(fwhm, npix = 128, size = 1, noise = 1000, ff = 10000, a = False, m
        = True):
    #define variables
    r3 = np.sqrt(3)
    std = fwhm/2.355 #convert fwhm to stdev
    r = npix//2
    fiber = makehex(r) #square mask to represent pixel
    center = npix//2

    #make grid of positions across pixel space
    x = np.linspace(-.5*size,.5*size,npix)
    y = np.linspace(-.5,.5,npix)
    pos = [[[x[i],y[j]] for j in range(npix)] for i in range(npix)]

    if m:
        tpsf = moffat((0,0), x, y, fwhm, 2.9)
        tpsf = tpsf/np.sum(tpsf)
    else:
        pdf = stats.multivariate_normal([0,0], [[std,0],[0,std]])
        tpsf = pdf.pdf(pos)

    tpsf = tpsf/np.sum(tpsf)
    psf = tpsf * ff
    

    #get the actual data
    data = convolve((center,center), fiber, psf)
    data = np.random.poisson(data) + np.random.normal(0, noise)

    #calculate error, s/n
    #doesn't use optext because it isn't necessary and breaks the function
    error = np.square(noise) + data #variance of noise
    sn = data/error
    return data, error, sn

#umbrella function for other opthexts, calls appropriate one
def opthext(fibers,fwhm,npix = 128,size = 1,noise = 1000,ff = 10000,a = False):
    if fibers == 1:
        return opthext0(fwhm, npix, size, noise, ff, a)
    elif fibers == 7:
        return opthext1(fwhm, npix, size, noise, ff, a)
    elif fibers == 19:
        return opthext2(fwhm, npix, size, noise, ff, a)
    else:
        print("Please put in a valid number of fibers (1,7, or 19)")
        exit()

#make a rectangle of 1s in a square of 0s
def makerect(w, h):
    if w > h:
        rect = np.zeros((w,w))
        for i in range(w):
            if i >= (w-h)/2 and i < (w+h)/2:
                rect[i,:] = 1
    else:
        rect = np.zeros((h,h))
        for i in range(h):
            if i >= (h-w)/2 and i < (w+h)/2:
                rect[:,i] = 1
    return rect

#do optimal extraction on a slit with multiple pixels 
#pixelsize given in fraction of total slit length
def optrext(pixelsize, fwhm, npix=100, noise=50, ff=10000, a=False, m=True,
        size = 1):
    #define variables
    width = .75 #width of slit in arcsec
    std = fwhm/2.355 #convert fwhm to stdev
    size = size #length of slit in arcsec
    pixheight = pixelsize*npix/size #height of pixel in terms of npix pixels

    #make a rectangular pixel mask to convolve over psf
    pixel = np.ones((int(pixelsize*npix/size), int(npix*width/size)))
    pixels = int(size/pixelsize)

    #make grid of positions across pixel space
    x = np.linspace(-.5*size,.5*size,npix)
    y = np.linspace(-.5*size,.5*size,npix)
    pos = [[[x[i],y[j]] for j in range(npix)] for i in range(npix)]

    #define 2d pdf, evaluate to make a grid of values, normalize
    if m:
        tpsf = moffat((0,0), x, y, fwhm, 2.9)
        tpsf = tpsf/np.sum(tpsf)
    else:
        pdf = stats.multivariate_normal([0,0], [[std,0],[0,std]])
        tpsf = pdf.pdf(pos)

    #make sure each has appropriate sum
    tpsf = tpsf/np.sum(tpsf)
    psf = tpsf * ff

    #convolve fibers with psfs to get simulated fiber data
    #(sorry about how convoluted these lines are, I kind of was just having fun
    #with how much I could do at once by abusing lots of stuff)
    #'''
    pixellocs = np.asarray(list(zip(np.linspace(0,npix,pixels+1) + pixheight/2,
        np.zeros(pixels+1)+npix//2))).astype(int)
    data = addnoise(np.random.poisson(np.asarray([convolve(pixellocs[i], 
        pixel, psf) for i in range(pixels)])), noise)
    tdata = np.asarray([convolve(pixellocs[i], pixel, tpsf)
        for i in range(pixels)])
    '''
    alldata = convolve2d(psf, pixel, mode = 'valid')
    talldata = convolve2d(tpsf, pixel, mode = 'valid')
    ypos = list(range(0,len(alldata)-int(pixheight),int(pixheight)))\
            + [len(alldata)-1]
    data = addnoise(np.random.poisson(alldata[ypos]), noise)
    tdata = talldata[ypos]
    '''
    var = np.square(noise) + data #variance of noise
    return optimalextract(data, tdata, var, aperture = a) 

#return a coordinate grid with ordered pairs of each x,y combination
#equivalent to [[[x[i],y[j]] for j in range(npix)] for i in range(npix)]
#barely even faster, but there's probably a more elegant way to do this
def makegrid(x, y):
    points = np.asarray(np.meshgrid(x, y, indexing = 'ij'))
    return np.swapaxes(np.swapaxes(points, 0, 1), 1, 2)

#computes amount of psf flux actually captured with a given aperture mask
def lossfrac(aperture, fwhm, m = False, psf = 'm', noise = 0, npix = 1000,
        size = 10, trunc = 0, n=1):
    x = np.linspace(-.5*size,.5*size,npix)
    y = np.linspace(-.5*size,.5*size,npix)
    
    if psf == 'm' or  m: #moffat
        tpsf = moffat((0,0), x, y, fwhm, 2.9)
    elif psf == 'g': #gaussian
        pos = [[[x[i],y[j]] for j in range(npix)] for i in range(npix)]
        pdf = stats.multivariate_normal([0,0], [[std,0],[0,std]])
        tpsf = pdf.pdf(pos)
    elif psf == 's': #sersic, in this case fwhm is actually n
        tpsf = sersic(x,y,fwhm,1)
    else:
        print("Input valid psf type (m for moffat, g for gaussian, s for\
        sersic)")
        return

    tpsf = tpsf/np.sum(tpsf)

    #truncate the psf after a certain radius (in arcsec) if desired
    if trunc:
        for xi,xs in enumerate(x):
            for yi,ys in enumerate(y):
                if np.sqrt(xs**2 + ys**2) > trunc:
                    tpsf[xi,yi] = 0

    return convolve((npix//2, npix//2), aperture, tpsf)

#takes an input array and adds a fiber array to it at position center
#fibers should be 0s and 1s (probably from makehex) or makerect
def addfiber(fiber, background, center):
    odd = (fiber.shape[0]%2, fiber.shape[1]%2) #accounts for odd dimensions
    background[center[0]-fiber.shape[0]//2:center[0]+fiber.shape[0]//2+odd[0], 
        center[1]-fiber.shape[1]//2:center[1]+fiber.shape[1]//2+odd[1]] += fiber
    return background

#puts together the appropriate fibers to make a bundle of specific hex fibers
def makebundle(fibers, fibersize, npix=1000, arcsec = 100):
    r3 = np.sqrt(3)
    r = int(fibersize * arcsec/r3)
    bundle = np.zeros((npix,npix))
    fiber = makehex(r)
    center = npix//2

    if fibers not in [1,7,19]:
        print("Please put in appropriate fiber count (1, 7, or 19)")
        return

    if fibers >= 1:
        bundle = addfiber(fiber, bundle, (center, center))

    if fibers >= 7:
        bundle = addfiber(fiber,bundle,(center, int(center+r*r3)))
        bundle = addfiber(fiber,bundle,(int(center+1.5*r), int(center+r*r3/2)))
        bundle = addfiber(fiber,bundle,(int(center+1.5*r), int(center-r*r3/2)))
        bundle = addfiber(fiber,bundle,(center, int(center-r*r3)))
        bundle = addfiber(fiber,bundle,(int(center-1.5*r), int(center-r*r3/2)))
        bundle = addfiber(fiber,bundle,(int(center-1.5*r), int(center+r*r3/2)))

    if fibers == 19:
        bundle = addfiber(fiber,bundle,(center, int(center+2*r*r3)))
        bundle = addfiber(fiber,bundle,(int(center+1.5*r),int(center+1.5*r*r3)))
        bundle = addfiber(fiber,bundle,(center+3*r, int(center+r*r3)))
        bundle = addfiber(fiber,bundle,(center+3*r, center))
        bundle = addfiber(fiber,bundle,(center+3*r, int(center-r*r3)))
        bundle = addfiber(fiber,bundle,(int(center+1.5*r),int(center-1.5*r*r3)))
        bundle = addfiber(fiber,bundle,(center, int(center-2*r*r3)))
        bundle = addfiber(fiber,bundle,(int(center-1.5*r),int(center-1.5*r*r3)))
        bundle = addfiber(fiber,bundle,(center-3*r, int(center-r*r3)))
        bundle = addfiber(fiber,bundle,(center-3*r, center))
        bundle = addfiber(fiber,bundle,(center-3*r, int(center+r*r3)))
        bundle = addfiber(fiber,bundle,(int(center-1.5*r), int(center+1.5*r*r3)))

    return bundle

def makecircle(r):
    r = int(r)
    circle = np.zeros((2*r, 2*r))
    for x in range(2*r):
        for y in range(2*r):
            if (x-r)**2 + (y-r)**2 < r**2:
                circle[x,y] = 1
    return circle

#return an elliptical mask with semimajor axis a, q = b/a, rotation t
def makeellipse(a, q, t=0, m = True):
    if abs(q) > 1 or 1 <= 0:
        raise Exception('q must be between 0 and 1')
    b = int(a*q)
    a = int(a)
    x,y = np.ogrid[-a:a, -a:a]
    ellipse = np.zeros((2*a, 2*a))
    if m:
        ellipse = np.ma.masked_values(ellipse, 0)
    mask = ((x*np.cos(t) + y*np.sin(t))/b)**2 + \
            ((x*np.sin(t) - y*np.cos(t))/a)**2 < 1
    ellipse[mask] = 1
    return ellipse

#shear a galaxy with q = b/a with even shear gp and odd shear gx
#from Huff et al.
def shear(q, gp, gx):
    ei = (1 - q**2)/(1 + q**2)
    eo = ei + 2* gp *(1 - ei**2)
    qo = np.sqrt((1 - eo)/(1 + eo))
    tr = gx/ei
    return qo, tr

#transform x and y into polar for a galaxy inclined at i and rotated at pa
def polar(x,y,i,pa):
    xd = x*np.sin(pa) + y*np.cos(pa)
    yd = (y*np.sin(pa) - x*np.cos(pa))/np.cos(i)
    r = np.sqrt(xd**2 + yd**2)
    theta = np.arctan2(-yd, xd)
    return r, theta

#make an arctan velocity field with radius size at inclination i 
#rotated at position angle pa with scale h and bulk vel b
def makevf(size, vmax, i, pa, h, b=0):
    vf = np.zeros((2*size, 2*size))
    x,y = np.ogrid[-size:size, -size:size]
    r, theta = polar(x,y,i,-pa)
    return vmax * np.tanh(r/h) * np.cos(theta) * np.sin(i) + b

#returns a sersic profile with a peak value of I, similar to moffat
#sersic index n is 1 for spirals, 4 for elliptical, a is 1/e radius
#formula from http://www.simondriver.org/Teaching/AS3011/AS3011_5.pdf
def sersic(x, y, n, a, I=1, center = None, inc = 0, pa = 0):
    if not center:
        center = [len(x)/2,len(y)/2]
    xi,yj = makexy(len(x))
    r, th = polar(xi-center[0], yj-center[1], *np.radians((inc,-pa)))
    return I * np.exp((-r/a)**(1/n))

#much more efficient than other moffat
def moffat2(x, y, fwhm, beta = 2.9, center = None, norm = True):
    alpha = fwhm/(2*np.sqrt(2**(1/beta)-1))
    if not center:
        center = (len(x)//2,len(y)//2)
    xi,yj = makexy(len(x))
    psf = dists.Moffat2D.evaluate(xi, yj, 1, x_0 = center[0],
            y_0 = center[1], gamma = alpha, alpha = beta)
    return psf

def makelensed(size, gp, gx, i, t, vmax, h, n=1, b=0):
    q = np.cos(i)
    qo, tr = shear(q, gp, gx)
    x,y = np.ogrid[-size:size, -size:size]
    galaxymask = makeellipse(size, qo, t+tr)
    vf = makevf(size, vmax, i, t + tr + vfrot(q, gx), h, b)
    print(tr, vfrot(q, gx))
    #vf = resize(makevf(size, vmax, i, t, h, b), (2*size,int(2*size*q)))
    #vfn = np.zeros((2*size,2*size))
    #vfn[:,size-vf.shape[1]//2:size+vf.shape[1]//2] += vf
    ser = sersic(x, y, n, h)
    return vf, galaxymask, ser

def vfrot(q,gx):
    ei = (1 - q**2)/(1 + q**2)
    return np.arcsin(-gx * (1+ei)/ei)

#return the shapes of many arrays, just saves some typing
def shapes(*arrays):
    return [a.shape for a in arrays]

#do a light weighted average of the pixels within a fiber
def weightedconvolve(center, fiber, psf, intensity):
    odd = (fiber.shape[0]%2, fiber.shape[1]%2) #accounts for odd dimensions
    xoffset = fiber.shape[1]//2
    yoffset = fiber.shape[0]//2

    #pick out the appropriate pixels in psf and intensity and do weighted avg
    data = (fiber * psf[center[0]-yoffset:center[0]+yoffset+odd[0],
            center[1]-xoffset:center[1]+xoffset+odd[1]])
    weights = (fiber * intensity[center[0]-yoffset:center[0]+yoffset+odd[0],
            center[1]-xoffset:center[1]+xoffset+odd[1]])
    return np.average(data, weights = weights)

#generate a velocity field of a galaxy and observe it with a fiber bundle
#plot vel field, intensity, and fiber data, return data
def vfobserve(vmax, i, h, pa = 0, fwhm = 5, noise = 50, size = 100, fmin = .01,
        fmax = .2, returnz=False, f19 = False):
    #parameters
    h = size/3
    if f19:
        size = int(size * 5/3)
    q = np.cos(np.radians(i))
    x = np.arange(0,2*size)
    center = size
    cen = (center,center)

    #make galaxy, velocity field, brightness map
    e = makeellipse(size, q)
    v = makevf(size, vmax, np.radians(i), np.radians(pa), h)
    b = sersic(x, x, 1, h, inc = i, pa = pa)
    b /= b.max()

    #make psf
    if fwhm:
        xp = np.arange(fwhm*4)
        psf = moffat2(np.arange(0,fwhm*4), np.arange(0,fwhm*4), fwhm)
        psf /= psf.sum()
    else:
        psf = [1]

    #convolve psf with velocity field and brightness to blur
    if noise:
        nv = cnoise(v, fmin, fmax) #generate lumpy noise for velocity field
        pv = convolve2d(nv, psf, mode = 'same', boundary = 'symm')
        b = convolve2d(b, psf, mode = 'same', boundary = 'symm')
        b /= b.max()

        #plot vel field and intensity
        plt.figure(figsize = (14,4))
        plt.subplot(131)
        plt.imshow(pv, cmap = 'RdBu')
        plt.colorbar()

        plt.subplot(132)
        plt.imshow(b, cmap = 'bone')
        plt.colorbar()

    else:
        pv = v

    #define fiber radius
    r = size//3
    if f19:
        r = size//5

    #define fiber and coordinates of fibers in bundle
    #goes counterclockwise from +x axis, radially outwards
    r3 = np.sqrt(3)
    fiber = makehex(r)
    coords = [(center,center), 
            (center, int(center+r*r3)), 
            (int(center+1.5*r), int(center+(r*r3)/2)), 
            (int(center+1.5*r), int(center-(r*r3)/2)),
            (center, int(center-r*r3)), 
            (int(center-1.5*r), int(center-(r*r3)/2)),
            (int(center-1.5*r), int(center+(r*r3)/2))]

    if f19:
        coords += [(center, int(center+2*r*r3)),
            (int(center+1.5*r),int(center+1.5*r*r3)),
            (center+3*r, int(center+r*r3)),
            (center+3*r, center),
            (center+3*r, int(center-r*r3)),
            (int(center+1.5*r),int(center-1.5*r*r3)),
            (center, int(center-2*r*r3)),
            (int(center-1.5*r),int(center-1.5*r*r3)),
            (center-3*r, int(center-r*r3)),
            (center-3*r, center),
            (center-3*r, int(center+r*r3)),
            (int(center-1.5*r), int(center+1.5*r*r3))]

    #do weighted convolution of each fiber to get vel measurement
    data = np.zeros(len(coords))
    z = np.zeros_like(pv)
    for i in range(len(coords)):
        c = coords[i]
        data[i] = weightedconvolve(c, fiber, pv, b)
        z = addfiber(data[i]*fiber, z, c)

    if noise:
        error = np.array([1] + [5]*(len(coords)-1))
        data = addnoise(data,error)

    #plot fiber bundle showing data
    if returnz:
        z = np.ma.array(z, mask = z==0)
        plt.subplot(133)
        plt.imshow(z, cmap = 'RdBu', vmin = data.min(), vmax = data.max())
        plt.colorbar()
    return data

#find difference between data and another vf, used in vfit
def vfdiff(guess, data, f19, error):
    if not error.all():
        error = np.ones_like(data)
    return (data - vfobserve(*guess, fwhm=0, noise=0, f19=f19))/(error)

#do least squares fitting of vmax, inc, hrot, and pa for a given data set
#from vfobserve, outputs plots, best fit, and errors
def vfit(data, f19=False, guess = None, error = False, plot = True):
    if not guess:
        guess = (100, 45, 20, 0)

    if error:
        try:
            len(error)
        except:
            error = [1] + [5]*6
            if f19:
                error += [5]*12
        error = np.array(error)

    fit = least_squares(vfdiff, guess, args = [data, f19, error])
    if plot:
        print('vmax, inc, hrot, pa')
        plt.tight_layout()
        plt.show()

    try:
        err = np.sqrt(np.diagonal(np.linalg.inv(np.dot(fit.jac.T,fit.jac))))
    except:
        print('Problems with error calculation')
        err = np.array([np.nan]*len(fit.x))

    return fit.x, err

#make spatially correlated noise by superimposing a bunch of 2D sine waves
#random spatial frequencies between fmin and fmax, adjust as needed
def cnoise(array, fmin, fmax, order = 100, amp = 1):
    x,y = makexy(len(array))
    z = np.zeros_like(x, dtype = float)

    for i in range(order):
        A = np.random.normal(amp, amp/3)
        a = np.radians(np.random.uniform(0, 90))
        xf = np.random.uniform(fmin, fmax)
        yf = np.random.uniform(fmin, fmax)
        xp = np.random.uniform(0,2*np.pi)
        yp = np.random.uniform(0,2*np.pi)
        z += A * (np.cos(xf*x*np.cos(a) - yf*y*np.sin(a) + xp) 
            + np.sin(xf*x*np.sin(a) + yf*y*np.cos(a) + yp))
    return array + z

#make a grid of x values along one axis, y along the other
def makexy(size):
    x = np.dstack([np.arange(size)] * size)[0]
    return x, x.T

