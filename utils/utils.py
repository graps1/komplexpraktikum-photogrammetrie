#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 16:59:33 2020

@author: hans
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl

font = {'family' : 'DejaVu Sans',
        'weight' : 'light',
        'size'   : 25}

mpl.rc('font', **font)

WV_COUNT = 200
    
def load_dataset(filepath):
    names=["id", "class", "x", "y", "z"] + [i for i in range(0,WV_COUNT)]
    data = pd.read_csv(filepath, sep=" ", header=None, names=names)
    # we need to invert z
    data["z"] = max(data["z"])-data["z"]
    return data

def waveforms_as_matrix(df):
    # this matrix will make it easier to plot samples
    mat = np.zeros((df.shape[0], WV_COUNT))
    for i in range(WV_COUNT):
        mat[:,i] = df[i].values
    return mat
        
def plot_waveforms(df, classcol="class", class_label_mapping={0:"land", 1:"water"}):
    # get waveforms in matrix form
    waveforms = waveforms_as_matrix(df)
    # set up figure
    fig = plt.figure(figsize=(23,7))
    ax = fig.add_subplot(111)
    ax.set_title("Waveforms")
    ax.set_xlabel("Index")
    ax.set_ylabel("Amplitude")
    # x-values
    xs = np.arange(WV_COUNT)
    # plot each sampled waveform
    occurred = set({})
    for idx in range(waveforms.shape[0]):
        # find the index in the waveform where it drops to 0 
        first_0_idx = np.where(waveforms[idx,:] == 0)[0]
        first_0_idx = first_0_idx[0] if len(first_0_idx != 0) else WV_COUNT 
        # the classifiaction (land or water) of this point
        dataclass = df[classcol][idx]
        # check if we already have a label for this class
        label = None
        if dataclass not in occurred:
            label = class_label_mapping[dataclass]
            occurred.add(dataclass)
        colors = ["g", "b", "r", "y", "b"]
        # plot full (colored) waveform (stripped of 0-elements)
        plt.plot(xs[:first_0_idx], 
                 waveforms[idx,:first_0_idx],
                 c=colors[dataclass],
                 label=label)
    plt.legend(loc='upper left')
    plt.show()

def plot_pcl(df, plotsize=20, classcol="class", xcol="x", ycol="y", zcol="z"):
    # initialize 3d-plot
    fig = plt.figure(figsize=(plotsize,plotsize))
    ax = fig.add_subplot(111, projection='3d')
    # get classes
    classes = pd.unique(df[classcol])
    # pts_by_class will contain rows sorted by their class
    pts_by_class = [ df[df[classcol] == cl ] for cl in classes ]
    for i,pts in enumerate(pts_by_class):
        # plot each point set which belongs to a specific class
        ax.scatter(pts[xcol], pts[ycol], pts[zcol], 
               # c=["g","b"][i], 
               marker=".", 
               # label=classes[i], 
               s=0.3)
    # ax.legend(loc=2)
    ax.grid(False)
    ax.set_axis_off()
    ax.set_xlim3d(-60,60)
    ax.set_ylim3d(-60,60)
    ax.set_zlim3d(-60,60)
    ax.view_init(elev=40., azim=-120.)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')