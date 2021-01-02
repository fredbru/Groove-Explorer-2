import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
import time
from MusicSOM.MusicSOM import *
import sys
import random
import pydeck
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource
from bokeh.models.tools import HoverTool

st.title('Groove Explorer 2')
palette_folder = 'Small_AL_Tester'
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
    colours = []

    for x, n, p, t in zip(features, names, palette_names, itemIndex):
        w = som.winner(x)
        weightMap[w] = im
        offsetX = random.uniform(-0.2, 0.2) #small x and y offsets to stop labels being plotted on top of each other
        offsetY = random.uniform(-0.2, 0.2)

        c='green'
        if p == 'Pop V3.bfd3pal':
            c = 'darkgreen'
        if p == 'Smooth Jazz.bfd3pal':
            c = 'darkmagenta'
        if p == 'Peter Erskine Rock.bfd3pal':
            c = 'dodgerblue'
        if p == 'Stanton Moore JB.bfd3pal':
            c = 'orangered'

        winners.append([n, p[:-8], w[0]+offsetX, w[1]+offsetY,c])
        im = im+1
    return winners

##
# Plot som output as labelled winner nodes, save output to csv file. Also plots U-Matrix underneath winner map.
def plot_winners_matplotlib(features, names, som, palette_names, showUMatrix=False):
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

def plot_winners_pydeck(features, names, som, palette_names):
    pass

def setup_SOM(palette_folder):
    combinedLabels = np.load(palette_folder + 'Names.npy')
    names = combinedLabels[0]
    #palette_names = os.listdir('/home/fred/BFD/python/grooves/' + sys.argv[1] + '/')
    palette_names = combinedLabels[1]
    features = np.load('Small_AL_Tester' + ".npy")
    features = features.astype(np.float32)
    featureLength = features.shape[1]

    som = MusicSOM(10, 10, featureLength, sigma=0.3, learning_rate=0.5, perceptualWeighting=False)
    som.random_weights_init(features)
    return som, features, names, palette_names


som, features, names, palette_names = setup_SOM(palette_folder)

with st.spinner('Generating SOM - please wait'):
    som.trainCPU(features, num_iterations=1000)
groove_mapping = pd.DataFrame(get_winners(features, names, som, palette_names), columns=['GrooveName',
                                                                'PaletteName', 'X', 'Y', 'Colour'])
source = ColumnDataSource(groove_mapping)
hover= HoverTool()
hover.tooltips=[
    ('Name', '@GrooveName'),
    ('Palette', '@PaletteName')
]

p = figure(title='Groove Explorer v2', x_range=(-1,10), y_range=(-1,10), tools=[])
p.add_tools(hover)
p.circle(source=source, x='X', y='Y', color='Colour',fill_alpha=0.6, size=10)
st.bokeh_chart(p, use_container_width=True)


