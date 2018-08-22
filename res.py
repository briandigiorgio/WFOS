#!/usr/bin/env python

from optexttools import *

if __name__ == '__main__':
    size = 100
    vmax = 100
    i = 60
    q = np.cos(np.radians(i))
    pa = 30
    h = 10
    x = np.arange(0,2*size)
    fwhm = 5
    center = size
    cen = (center,center)
    noise = 50

    e = makeellipse(size, q)
    v = makevf(size, vmax, np.radians(i), np.radians(pa), h)
    b = sersic(x,x,1,20)
    b = b/b.max()
    plt.subplot(133)
    plt.imshow(b)
    plt.colorbar()

    psf = moffat(cen, np.arange(0,fwhm*4), np.arange(0,fwhm*4), 
            fwhm, norm = True)
    nv = addnoise(v, noise)
    pv = convolve2d(nv, psf, mode = 'same', boundary = 'symm')

    plt.subplot(131)
    plt.imshow(pv, cmap = 'RdBu')
    plt.colorbar()

    r = size//3
    r3 = np.sqrt(3)
    fiber = makehex(r)
    coords = [(center,center), 
            (center, int(center+r*r3)), 
            (int(center+1.5*r), int(center+(r*r3)/2)), 
            (int(center+1.5*r), int(center-(r*r3)/2)),
            (center, int(center-r*r3)), 
            (int(center-1.5*r), int(center-(r*r3)/2)),
            (int(center-1.5*r), int(center+(r*r3)/2))]
    data = np.zeros(len(coords))
    z = np.zeros_like(pv)

    for i in range(len(coords)):
        c = coords[i]
        data[i] = weightedconvolve(c, fiber, pv, b)
        z  = drawhex(c, data[i]*fiber, r, z)
    plt.subplot(132)
    z = np.ma.array(z, mask = z==0)
    plt.imshow(z, cmap = 'RdBu', vmin = data.min(), vmax = data.max())
    plt.colorbar()

    print(data)
    plt.show()
