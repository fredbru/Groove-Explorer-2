import numpy as np
import pandas as pd
import time
from MusicSOM.MusicSOM import *
import sys
import random
import pydeck
from bokeh.layouts import column
from bokeh.plotting import figure, output_file, show, curdoc
from bokeh.models import ColumnDataSource, Button, TextInput
from bokeh.models.tools import HoverTool, TapTool, PointDrawTool
from bokeh.models.callbacks import CustomJS
from bokeh.events import PanEnd
from bokeh.models import Button
from bokeh.io import curdoc
from sklearn import preprocessing


# To run: bokeh serve --show run-bokeh.py

palette_folder = ''

def make_node_locations(dim):
    xNodePoints = np.arange(dim+1)
    yNodePoints = np.arange(dim+1)
    nodePoints = np.vstack([xNodePoints,yNodePoints])
    return nodePoints

def get_winners(features, names, som, palette_names):
    itemIndex = range(len(names))
    weightMap = {}
    im = 0
    winners = []
    colours = []

    for x, n, p, t in zip(features, names, palette_names, itemIndex):
        w = som.winner(x)
        weightMap[w] = im
        offsetX = round(random.uniform(-0.15, 0.15),2) #small x and y offsets to stop labels being plotted on top of each other
        offsetY = round(random.uniform(-0.15, 0.15),2)

        c='green'
        if p == 'Pop V3.bfd3pal':
            c = 'darkgreen'
        if p == 'Smooth Jazz.bfd3pal':
            c = 'navy'
        if p == 'Peter Erskine Rock.bfd3pal':
            c = 'dodgerblue'
        if p == 'Stanton Moore JB.bfd3pal':
            c = 'orangered'
        if p == 'AFJ Rock.bfd3pal':
            c = 'blueviolet'
        if p == 'Reggae Grooves V2.bfd3pal':
            c = 'chocolate'
        if p == 'HHM Jungle V1.bfd3pal':
            c = 'goldenrod'
        if p == 'Early RnB.bfd3pal':
            c = 'red'
        if p == 'Peter Erskine Jazz.bfd3pal':
            c = 'slategray'
        if p == 'Chicago Blues.bfd3pal':
            c = 'yellowgreen'
        if p == 'Steve Ferrone Rock V1.bfd3pal':
            c = 'skyblue'
        if p == 'Brooks Punk V1.bfd3pal':
            c = 'black'


        winners.append([n, p[:-8], w[0]+offsetX, w[1]+offsetY,c])
        im = im+1
    return winners

def setup_SOM(palette_folder, dim):
    combinedLabels = np.load('Mixed_8x12' + 'Names.npy')
    names = combinedLabels[0]
    #palette_names = os.listdir('/home/fred/BFD/python/grooves/' + sys.argv[1] + '/')
    palette_names = combinedLabels[1]
    features = np.load('Mixed_8x12' + ".npy")
    features = features.astype(np.float32)
    a = features
    # b = (a - np.min(a)) / np.ptp(a) #this doesn't seem to work...?
    # features = b
    scaler = preprocessing.MinMaxScaler()
    scaler.fit(a)
    features = scaler.transform(a)
    featureLength = features.shape[1]
    som = MusicSOM(dim, dim, featureLength, sigma=2.0, learning_rate=0.5, perceptualWeighting=False)
    som.random_weights_init(features)
    return som, features, names, palette_names

np.set_printoptions(suppress=True, precision=6)
np.set_printoptions(edgeitems=50, linewidth=100000)

dim = 12
som, features, names, palette_names = setup_SOM(palette_folder,dim)
print(features)
print("\n \n")
som.trainCPU(features, num_iterations=30000)
groove_map_info = pd.DataFrame(get_winners(features, names, som, palette_names), columns=['GrooveName',
                                                                'PaletteName', 'X', 'Y', 'Colour'])

print(som.weights)
print(som.weights.shape)
np.save("SOM_Weights.npy", som.weights)