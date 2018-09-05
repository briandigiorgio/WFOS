#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-n', '--num', type = int, help = 'number of lines',
            default = 12)
    parser.add_argument('-x', '--xkcd', nargs = 3, 
            help = 'parameters for plt.xkcd()', default = [1.5,90,5])
    args = parser.parse_args()

    plt.xkcd(*args.xkcd)
    plt.figure(figsize = (11,8.5))
    colors = ['k', 'lightcoral','maroon', 'r', 'coral', 'orangered', 'chocolate', 'orange', 'gold', 'chartreuse', 'limegreen', 'g', 'springgreen', 'aquamarine', 'c', 'dodgerblue', 'deepskyblue', 'b', 'blueviolet', 'mediumorchid', 'darkviolet', 'm', 'deeppink', 'fuchsia']
    locs = np.linspace(100,0,args.num)
    c = np.random.choice(colors, args.num, replace = False)

    for i in range(args.num):
        plt.axhline(locs[i], color = c[i])
    plt.axvline(.2, color = 'k')

    plt.xlim((0,1))
    plt.ylim((-5,105))
    plt.axis('off')
    plt.tight_layout()
    plt.show()

