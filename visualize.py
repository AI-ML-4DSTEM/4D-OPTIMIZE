import sys
import h5py
import numpy as np
from itertools import product
from time import sleep
import matplotlib.pyplot as plt

def plot_image(inputs, grid_size = [1,2], figsize = 20, show_axis = False):
    assert(type(inputs) == list)
    assert(type(figsize) == int or tuple)
    assert(type(grid_size) == list)
    assert(len(inputs) == grid_size[0]*grid_size[1])
    
    if type(figsize) == int:
        figsize = (figsize, figsize)
        
    fig, axs = plt.subplots(grid_size[0], grid_size[1], figsize = figsize)
    
    counter = -1
    if grid_size[0] == 1:
        for i in range(grid_size[1]):
            counter += 1
            axs[i].imshow(np.cbrt(inputs[counter]))
            if not show_axis:
                axs[i].axes.xaxis.set_visible(False)
                axs[i].axes.yaxis.set_visible(False)
    else:
        for i in range(grid_size[0]):
            for j in range(grid_size[1]):
                counter += 1
                axs[i][j].imshow(np.cbrt(inputs[counter]))
                if not show_axis:
                    axs[i][j].axes.xaxis.set_visible(False)
                    axs[i][j].axes.yaxis.set_visible(False)

    


def plot_line(x = None,y = None, xlabel = None, ylabel = None, figsize = 20, labelsize = 30, fontsize=30):
    assert(type(figsize) == int or tuple)
    assert(type(x) == np.ndarray or None)
    assert(type(y) == np.ndarray or list)
    
    if type(figsize) == int:
        figsize = (figsize, figsize)
        
    fig = plt.figure(figsize=figsize)

    if x is None:
        for i in range(len(y)):
            plt.plot(y[i])
    else:
        for i in range(len(y)):
            plt.plot(x, y[i])

    plt.tick_params(axis='both', which='major', labelsize=labelsize)
    if xlabel is not None:
        plt.xlabel(xlabel, fontsize = fontsize)
    if ylabel is not None:
        plt.ylabel(ylabel, fontsize = fontsize)
