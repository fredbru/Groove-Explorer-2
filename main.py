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

palette_folder = 'Groove-Explorer-2/'

def text_input_handler(attr, old, new):
    try:
        num_iterations = int(new)
    except ValueError:
        print("Please enter a valid number")

def pan_python_callback():

    for i in range(94):
        if source.data['X'][i] != groove_map_info['X'][i]:
            old_coordinates = [int(round(groove_map_info['X'][i],0)),
                               int(round(groove_map_info['Y'][i],0))]

            new_X = round(source.data['X'][i], 0) + round(random.uniform(-0.2, 0.2), 2)
            new_Y = round(source.data['Y'][i], 0) + round(random.uniform(-0.2, 0.2), 2)
            #source.patch({'X': [(i, new_X)], 'Y': [(i, new_Y)]})
            groove_map_info.at[i, 'X'] = new_X
            groove_map_info.at[i, 'Y'] = new_Y

            new_coordinates = [int(round(new_X, 0)), int(round(new_Y, 0))]
            print("Groove = ", source.data['GrooveName'][i])
            print("New coordinates = ", new_coordinates)
            print("Old coordinates =", old_coordinates)

            groove = features[i]
            som.update_active_learning_nurnberger_local(groove, new_coordinates, old_coordinates)
    groove_map_info.update(pd.DataFrame(get_winners(features, names, som, palette_names),
                                                columns=['GrooveName', 'PaletteName', 'X', 'Y', 'Colour']))
    for i in range(94):
        new_X = groove_map_info['X'][i]
        new_Y = groove_map_info['Y'][i]
        source.patch({'X': [(i, new_X)], 'Y': [(i, new_Y)]})
    print('Done')

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
    combinedLabels = np.load('Groove-Explorer-2/Mixed_8x12' + 'Names.npy')
    names = combinedLabels[0]
    #palette_names = os.listdir('/home/fred/BFD/python/grooves/' + sys.argv[1] + '/')
    palette_names = combinedLabels[1]
    features = np.load('Groove-Explorer-2/Mixed_8x12' + ".npy")
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

def regenerate_SOM(num_iterations=20):
    print('Regenerating SOM with ' + str(num_iterations) + " iterations.....")
    #som, features, names, palette_names = setup_SOM(palette_folder)
    som.trainCPU(features, num_iterations)
    groove_map_info.update(pd.DataFrame(get_winners(features, names, som, palette_names),
                                        columns=['GrooveName', 'PaletteName', 'X', 'Y','Colour']))
    for i in range(94):
        new_X = groove_map_info['X'][i]
        new_Y = groove_map_info['Y'][i]
        source.patch({'X': [(i, new_X)], 'Y': [(i, new_Y)]})
    print('Done')

TAPCODE = """
var alldata = source.data;
var selected = source.selected.indices;
console.log(selected)
var groovename = alldata['GrooveName'][selected[0]];
var filetype = ".mp3";
var directory = "Groove-Explorer-2/static/Mixed-8x12-MP3/";
var file = directory.concat(groovename, filetype);
console.log(groovename)
if (groovename != "undefined")
{
    var audio = new Audio(file);
    audio.play();
    console.log("Playing " + groovename + "...");
    }
"""
np.set_printoptions(suppress=True, precision=6)
np.set_printoptions(edgeitems=50, linewidth=100000)

dim = 12
som, features, names, palette_names = setup_SOM(palette_folder,dim)
print(features)
print("\n \n")
som.trainCPU(features, num_iterations=10)
som.weights = np.load("Groove-Explorer-2/SOM_Weights.npy")
groove_map_info = pd.DataFrame(get_winners(features, names, som, palette_names), columns=['GrooveName',
                                                                'PaletteName', 'X', 'Y', 'Colour'])
source = ColumnDataSource(groove_map_info)

hover= HoverTool()
hover.tooltips=[
    ('Name', '@GrooveName'),
    ('Palette', '@PaletteName'),
]

TOOLS = "crosshair, pan, wheel_zoom"
explorer = figure(x_range
                  =(-1,dim), y_range=(-1,dim), tools=TOOLS, title='Groove Explorer 2')
#explorer.image(image=[1.0-som.distance_map()], x=-0.5,y=-0.5, dw=dim, dh=dim, palette="Greys256", level='image')
tap = TapTool()
tap.callback=CustomJS(code=TAPCODE, args=dict(source=source))
explorer.add_tools(hover)
explorer.add_tools(tap)

renderer = explorer.circle(source=source, x='X', y='Y', color='Colour',fill_alpha=0.6, size=15,
         hover_fill_color='yellow', hover_alpha=1, nonselection_alpha=0.6)

point_drag = PointDrawTool(renderers=[renderer], add=False)
explorer.add_tools(point_drag)
explorer.on_event(PanEnd, pan_python_callback)

text_input = TextInput(value="1000", title="Number of iterations (Press Enter to generate):")
text_input.on_change("value", text_input_handler)



bt = Button(label='Add 500 trainings')

def change_click():
    regenerate_SOM(num_iterations=100)


bt.on_click(change_click)

curdoc().add_root(column(explorer,text_input, bt))
