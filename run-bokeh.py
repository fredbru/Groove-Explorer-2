import numpy as np
import pandas as pd
import time
from MusicSOM.MusicSOM import *
import sys
import random
import pydeck
from bokeh.layouts import column
from bokeh.plotting import figure, output_file, show, curdoc
from bokeh.models import ColumnDataSource, Button
from bokeh.models.tools import HoverTool, TapTool, PointDrawTool
from bokeh.models.callbacks import CustomJS
from bokeh.events import PanEnd
from functools import partial

output_file("tool.html")
palette_folder = 'Small_AL_Tester'
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
        # offsetX = round(random.uniform(-0.1, 0.1),1) #small x and y offsets to stop labels being plotted on top of each other
        # offsetY = round(random.uniform(-0.1, 0.1),1)

        c='green'
        if p == 'Pop V3.bfd3pal':
            c = 'darkgreen'
        if p == 'Smooth Jazz.bfd3pal':
            c = 'darkmagenta'
        if p == 'Peter Erskine Rock.bfd3pal':
            c = 'dodgerblue'
        if p == 'Stanton Moore JB.bfd3pal':
            c = 'orangered'

        winners.append([n, p[:-8], w[0], w[1],c])
        im = im+1
    return winners

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
        offsetX = random.uniform(-0.3, 0.3) #small x and y offsets to stop labels being plotted on top of each other
        offsetY = random.uniform(-0.3, 0.3)
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
som.trainCPU(features, num_iterations=1000)
groove_map_info = pd.DataFrame(get_winners(features, names, som, palette_names), columns=['GrooveName',
                                                                'PaletteName', 'X', 'Y', 'Colour'])

source = ColumnDataSource(groove_map_info)

TAPCODE = """
var alldata = source.data;
var selected = source.selected.indices;
var groovename = alldata['GrooveName'][selected[0]];
var filetype = ".mp3";
var directory = "Audio/";
var file = directory.concat(groovename, filetype);

console.log("Playing " + file + "...");
var audio = new Audio(file);
audio.play();
"""

DRAGCODE = """
console.log(cb_data);
var alldata = source.data;
var xdata = alldata['X'];

for (var i=0; i < xdata.length; i++) {
        console.log(xdata[i] + " " + cb_obj.x);
        if (xdata[i] == cb_obj.x)
            console.log(xdata[i]);
        }
"""

def pan_python_callback():
    for i in range(123):
        if source.data['X'][i] != groove_map_info['X'][i]:
            new_X = round(source.data['X'][i],0)
            new_Y = round(source.data['Y'][i],0)
            source.patch({'X': [(i, new_X)], 'Y': [(i, new_Y)]})
            groove_map_info.at[i, 'X'] = new_X
            groove_map_info.at[i, 'Y'] = new_Y


hover= HoverTool()
hover.tooltips=[
    ('Name', '@GrooveName'),
    ('Palette', '@PaletteName'),
    ('X', '@X')
]

TOOLS = "crosshair, pan, wheel_zoom"
p = figure(x_range=(-1,10), y_range=(-1,10), tools=TOOLS, title='Groove Explorer 2')

tap = TapTool()
tap.callback=CustomJS(code=TAPCODE, args=dict(source=source))
# p.js_on_event('tap', CustomJS(code=CODE, args=dict(audioname="test-audio.mp3")))
p.add_tools(hover)
p.add_tools(tap)

renderer = p.circle(source=source, x='X', y='Y', color='Colour',fill_alpha=0.6, size=13,
         hover_fill_color='yellow', hover_alpha=1, nonselection_alpha=0.6)


point_drag = PointDrawTool(renderers=[renderer], add=False)
p.add_tools(point_drag)
#p.js_on_event(events.PanEnd, CustomJS(code=DRAGCODE, args=dict(source=source)))
p.on_event(PanEnd, pan_python_callback)

# b = Button()
# b.on_click(lambda: print("CLICK!"))
curdoc().add_root(column(p))
