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
from functools import partial

# To run: bokeh serve --show run-bokeh.py

palette_folder = 'Groove-Explorer-2/Small_AL_Tester'
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
            c = 'darkmagenta'
        if p == 'Peter Erskine Rock.bfd3pal':
            c = 'dodgerblue'
        if p == 'Stanton Moore JB.bfd3pal':
            c = 'orangered'

        winners.append([n, p[:-8], w[0]+offsetX, w[1]+offsetY,c])
        im = im+1
    return winners

def setup_SOM(palette_folder):
    combinedLabels = np.load(palette_folder + 'Names.npy')
    names = combinedLabels[0]
    #palette_names = os.listdir('/home/fred/BFD/python/grooves/' + sys.argv[1] + '/')
    palette_names = combinedLabels[1]
    features = np.load('Groove-Explorer-2/Small_AL_Tester' + ".npy")
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
var directory = "Groove-Explorer-2/static/Audio/";
var file = directory.concat(groovename, filetype);


console.log("Playing " + groovename + "...");
var audio = new Audio(file);
audio.play();
"""

def pan_python_callback():
    for i in range(123):
        if source.data['X'][i] != groove_map_info['X'][i]:
            new_X = round(source.data['X'][i],0) + round(random.uniform(-0.15, 0.15),2)
            new_Y = round(source.data['Y'][i],0) + round(random.uniform(-0.15, 0.15),2)
            source.patch({'X': [(i, new_X)], 'Y': [(i, new_Y)]})
            groove_map_info.at[i, 'X'] = new_X
            groove_map_info.at[i, 'Y'] = new_Y

def regenerate_SOM(num_iterations=1000):
    print('Regenerating SOM with ' + str(num_iterations) + " iterations.....")
    som, features, names, palette_names = setup_SOM(palette_folder)
    som.trainCPU(features, num_iterations)
    groove_map_info.update(pd.DataFrame(get_winners(features, names, som, palette_names),
                                        columns=['GrooveName', 'PaletteName', 'X', 'Y','Colour']))
    for i in range(123):
        new_X = groove_map_info['X'][i]
        new_Y = groove_map_info['Y'][i]
        source.patch({'X': [(i, new_X)], 'Y': [(i, new_Y)]})
    print('Done')

def text_input_handler(attr, old, new):
    try:
        num_iterations=int(new)
        regenerate_SOM(num_iterations=num_iterations)
    except ValueError:
        print("Please enter a valid number")

hover= HoverTool()
hover.tooltips=[
    ('Name', '@GrooveName'),
    ('Palette', '@PaletteName'),
]

TOOLS = "crosshair, pan, wheel_zoom"
explorer = figure(x_range=(-1,10), y_range=(-1,10), tools=TOOLS, title='Groove Explorer 2')

tap = TapTool()
tap.callback=CustomJS(code=TAPCODE, args=dict(source=source))
explorer.add_tools(hover)
explorer.add_tools(tap)

renderer = explorer.circle(source=source, x='X', y='Y', color='Colour',fill_alpha=0.6, size=13,
         hover_fill_color='yellow', hover_alpha=1, nonselection_alpha=0.6)

point_drag = PointDrawTool(renderers=[renderer], add=False)
explorer.add_tools(point_drag)
#p.js_on_event(events.PanEnd, CustomJS(code=DRAGCODE, args=dict(source=source)))
explorer.on_event(PanEnd, pan_python_callback)

text_input = TextInput(value="1000", title="Number of iterations (Press Enter to generate):")
text_input.on_change("value", text_input_handler)

curdoc().add_root(column(explorer,text_input))
