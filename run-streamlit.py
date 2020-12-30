import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
import time
from MusicSOM.MusicSOM import *
import sys
from matplotlib import pyplot as plt
import random


st.title('Groove Explorer 2')
palette_names = 'Small_AL_Tester'
def make_node_locations(dim):
    xNodePoints = np.arange(dim+1)
    yNodePoints = np.arange(dim+1)
    nodePoints = np.vstack([xNodePoints,yNodePoints])
    return nodePoints

##
# Get winners of SOM without plotting - for use when running on servers without matplotlib support.
def get_winners(features, names, som, palette_names):
    itemIndex = range(len(names))
    weightMap = {}
    im = 0
    winners = []
    for x, g, p, t in zip(features, names, palette_names, itemIndex):
        w = som.winner(x)
        weightMap[w] = im
        winners.append([g, p, w[0], w[1]])
        im = im+1
    return winners

##
# Plot som output as labelled winner nodes, save output to csv file. Also plots U-Matrix underneath winner map.
def plot_winners(features, names, som, palette_names, showUMatrix=False):
    plt.figure()
    itemIndex = range(len(names))
    weightMap = {}
    im = 0
    winners = []
    previousPalette = 'none'
    if showUMatrix == True:
        plt.pcolor(som.distance_map().T)
    for x, g, p, t in zip(features, names, palette_names, itemIndex):  # scatterplot
        w = som.winner(x)
        weightMap[w] = im
        offsetX = random.uniform(-0.2, 0.2) #small x and y offsets to stop labels being plotted on top of each other
        offsetY = random.uniform(-0.2, 0.2)
        if p != previousPalette:
            colour = np.random.uniform(low=0.1,high=0.9, size=3)
            print(colour)
        plt.text(w[0] + offsetX +.5, w[1] + offsetY +.5, g,
                 color=colour, fontdict={'size': 7})

        #plt.annotate(l, (w[0] + offset, w[1]+offset))
        winners.append([g, p, w[0],w[1]])
        im = im + 1
        previousPalette = p
    plt.axis([0, som.weights.shape[0], 0, som.weights.shape[1]])
    nodes = np.indices((som.x,som.y)).reshape(2,-1)
    nx = list(nodes[0])
    nxOffset = [x+0.5 for x in nx]
    ny = list(nodes[1])
    nyOffset = [x+0.5 for x in ny]
    plt.scatter(nxOffset, nyOffset, 3) #plots SOM node positions as blue dots on grid
    plt.show(block=False)
    return winners

def setup_SOM(palette_names):
    combinedLabels = np.load(palette_names + 'Names.npy')
    names = combinedLabels[0]
    #palette_names = os.listdir('/home/fred/BFD/python/grooves/' + sys.argv[1] + '/')
    palette_name = combinedLabels[1]
    print(palette_name)
    features = np.load('Small_AL_Tester' + ".npy")
    features = features.astype(np.float32)
    featureLength = features.shape[1]

    som = MusicSOM(10, 10, featureLength, sigma=0.3, learning_rate=0.5, perceptualWeighting=False)
    som.random_weights_init(features)
    return som, features, names


latest_iteration = st.empty()
bar = st.progress(0)

som, features, names = setup_SOM(palette_names)

with st.spinner('Generating SOM - please wait'):
    som.trainCPU(features, num_iterations=1000)
winners = get_winners(features, names, som, palette_names)
st.success('Done!')

plot_winners(features, names, som, palette_names)
plt.show()
