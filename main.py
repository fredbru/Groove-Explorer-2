import numpy as np
import pandas as pd
from MusicSOM.MusicSOM import *
import random
from bokeh.layouts import column, row
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, Panel, Tabs, RadioGroup, MultiSelect, RadioButtonGroup
from bokeh.models.tools import HoverTool, TapTool, PointDrawTool
from bokeh.models.callbacks import CustomJS
from bokeh.events import PanEnd
from bokeh.models import Button
from bokeh.io import curdoc
from sklearn import preprocessing
import os


np.set_printoptions(suppress=True, precision=6)
np.set_printoptions(edgeitems=50, linewidth=100000)

# To run: bokeh serve --show run-bokeh.py

palette_folder = 'Groove-Explorer-2/'

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
        offsetX = round(random.uniform(-0.27, 0.27),2) #small x and y offsets to stop labels being plotted on top of each other
        offsetY = round(random.uniform(-0.27, 0.27),2)

        c='white'
        if p == 'Pop V3.bfd3pal':
            c = 'darkgreen'
        if p == 'Smooth Jazz.bfd3pal':
            c = 'gold'
        if p == 'Peter Erskine Rock.bfd3pal':
            c = 'lightblue'
        if p == 'Stanton Moore JB.bfd3pal':
            c = 'red'
        if p == 'AFJ Rock.bfd3pal':
            c = 'steelblue'
        if p == 'Reggae Grooves V2.bfd3pal':
            c = 'black'
        if p == 'HHM Jungle V1.bfd3pal':
            c = 'blueviolet'
        if p == 'Early RnB.bfd3pal':
            c = 'brown'
        if p == 'Peter Erskine Jazz.bfd3pal':
            c = 'orange'
        if p == 'Chicago Blues.bfd3pal':
            c = 'chocolate'
        if p == 'Steve Ferrone Rock V1.bfd3pal':
            c = 'midnightblue'
        if p == 'Brooks Punk V1.bfd3pal':
            c = 'deepskyblue'

        if p == 'Funk V1.bfd3pal':
            c = 'salmon'
        if p == 'HHM Jungle V2.bfd3pal':
            c = 'mediumslateblue'
        if p == 'HHM Rave.bfd3pal':
            c = 'darkmagenta'
        if p == 'Jazz Walk Sticks.bfd3pal':
            c = 'darkkhaki'
        if p == 'Essential Swing.bfd3pal':
            c = 'darkorange'
        if p == 'Pop V1.bfd3pal':
            c = 'mediumseagreen'
        if p == 'Blues.bfd3pal':
            c = 'maroon'
        if p == 'Texas Blues.bfd3pal':
            c = 'sienna'
        if p == 'Essential Alternative Rock.bfd3pal':
            c = 'darkturquoise'
        if p == 'Bobby Jarzombek Rock.bfd3pal':
            c = 'navy'
        if p == 'Reggae Grooves V3.bfd3pal':
            c = 'slategrey'
        if p == 'Hard Rock.bfd3pal':
            c = 'cornflowerblue'

        if p == 'Cha Cha.bfd3pal':
            c = 'silver'
        if p == 'Funk V3.bfd3pal':
            c = 'indianred'
        if p == 'Soul Grooves.bfd3pal':
            c = 'firebrick'
        if p == 'Top 30.bfd3pal':
            c = 'purple'
        if p == 'New Country V2.bfd3pal':
            c = 'tan'
        if p == 'Swing Jazz.bfd3pal':
            c = 'yellow'
        if p == 'Soul Blues.bfd3pal':
            c = 'saddlebrown'
        if p == 'Glam Get Down.bfd3pal':
            c = 'lightgreen'
        if p == 'Trash Metal.bfd3pal':
            c = 'dodgerblue'
        if p == 'Steve Ferrone Rock V2.bfd3pal':
            c = 'mediumblue'
        if p == 'Jazz Brushes V2.bfd3pal':
            c = 'orangered'
        if p == 'HHM Jungle V3.bfd3pal':
            c = 'magenta'

        winners.append([n, p[:-8], w[0]+offsetX, w[1]+offsetY,c])
        im = im+1
    return winners

def setup_SOM(data, dim):
    path = 'Groove-Explorer-2/'
    combinedLabels = np.load(path+data + 'Names.npy')
    coefficients_file = data + 'Coefficients.npy'
    names = combinedLabels[0]

    palette_names = combinedLabels[1]
    features = np.load(path+data + ".npy")
    features = features.astype(np.float32)
    a = features
    scaler = preprocessing.MinMaxScaler()
    scaler.fit(a)
    features = scaler.transform(a)
    featureLength = features.shape[1]
    som = MusicSOM(dim, dim, featureLength, sigma=1.5, learning_rate=0.5, perceptualWeighting=False,
                   coefficients_file=coefficients_file)
    som.random_weights_init(features)
    return som, features, names, palette_names

def make_explorer(data_file, explorer_type='Customised'):
    #explorer_type options: 'Customised', 'Small', 'Big'

    def go_back():
        som.revert_active_learning()
        groove_map_info.update(pd.DataFrame(get_winners(features, names, som, palette_names),
                                            columns=['GrooveName', 'PaletteName', 'X', 'Y', 'Colour']))
        for i in range(94):
            new_X = groove_map_info['X'][i]
            new_Y = groove_map_info['Y'][i]
            source.patch({'X': [(i, new_X)], 'Y': [(i, new_Y)]})

    def pan_python_callback():
        print('pan callback executed')
        for i in range(94):
            if source.data['X'][i] != groove_map_info['X'][i]:
                old_coordinates = [int(round(groove_map_info['X'][i], 0)),
                                   int(round(groove_map_info['Y'][i], 0))]

                new_X = round(source.data['X'][i], 0) + round(random.uniform(-0.2, 0.2), 2)
                new_Y = round(source.data['Y'][i], 0) + round(random.uniform(-0.2, 0.2), 2)
                # source.patch({'X': [(i, new_X)], 'Y': [(i, new_Y)]})
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

    def make_audio_panel(explorer_type):
        PLAY_TEST_AUDIO = """
        var index = selector.active;
        var labels = ['A', 'B', 'C', 'D', 'E'];
        var filename = path + labels[index] + '.mp3';
        audio_player.stop_audio();
        audio_player.make_audio(filename);
        audio_player.play_audio();
        """

        STOPCODE = """
        audio_player.stop_audio();
        """

        if explorer_type == 'Small':
            test_audio_path = 'Groove-Explorer-2/static/Test Audio/Groove Explorer Part 1 - Small/'
        if explorer_type == 'Customised':
            test_audio_path = 'Groove-Explorer-2/static/Test Audio/Groove Explorer Part 2 - Customisable/'

        labels = ['A', 'B', 'C', 'D', 'E']
        audio_selector = RadioGroup(labels=labels, height_policy="auto", sizing_mode='scale_width', active=0)

        play_button = Button(label='Play')
        play_button.js_on_click(CustomJS(args=dict(selector=audio_selector, path=test_audio_path), code=PLAY_TEST_AUDIO))
        stop_button = Button(label='Stop')
        stop_button.js_on_click(CustomJS(code=STOPCODE))
        audio_panel = column(audio_selector, play_button, stop_button)
        return audio_panel

    PLAY_FROM_EXPLORER = """
    var selected = source.selected.indices;
    var groovename = source.data['GrooveName'][selected[0]];
    var palette = source.data['PaletteName'][selected[0]];
    var filetype = ".mp3";
    var filename = path + palette + '/' + groovename + '.mp3'
    audio_player.stop_audio();
    audio_player.make_audio(filename);
    audio_player.play_audio();
    audio.play(); """

    hover = HoverTool()
    hover.tooltips = [
        ('Name', '@GrooveName'),
        ('Palette', '@PaletteName'),
    ]
    TOOLS = "crosshair, wheel_zoom, pan, reset"

    if explorer_type in ['Small','Customised']:
        dim = 12
        som, features, names, palette_names = setup_SOM(data_file, dim)
        if explorer_type == 'Small':
            explorer_audio_path = "Groove-Explorer-2/static/Part 1 MP3 - Seperate Folders/"
            som.weights = np.load("Groove-Explorer-2/SOM_Weights_MLR_3M_Part1.npy")
        elif explorer_type == 'Customised':
            explorer_audio_path = "Groove-Explorer-2/static/Part 3 MP3 - Seperate Folders/"
            som.weights = np.load("Groove-Explorer-2/SOM_Weights_MLR_2M_Part3.npy")
        groove_map_info = pd.DataFrame(get_winners(features, names, som, palette_names),
                                       columns=['GrooveName','PaletteName', 'X', 'Y','Colour'])
        source = ColumnDataSource(groove_map_info)
        explorer = figure(x_range=(-1, dim), y_range=(-1, dim), tools=TOOLS, title='Groove Explorer 2')
        audio_selector = make_audio_panel(explorer_type)


    elif explorer_type == 'Big':
        dim = 28
        som, features, names, palette_names = setup_SOM(data_file, dim)
        explorer_audio_path = "Groove-Explorer-2/static/Big_Dataset_Reduced/"
        som.weights = np.load("Groove-Explorer-2/SOM_Weights_MLR_3M_BIG_s3-5_28x28.npy")
        groove_map_info = pd.DataFrame(get_winners(features, names, som, palette_names),
                                       columns=['GrooveName', 'PaletteName', 'X', 'Y', 'Colour'])
        source = ColumnDataSource(groove_map_info)
        explorer = figure(x_range=(-1, dim), y_range=(-1, dim), tools=TOOLS,
                          title='Groove Explorer 2', plot_width=700, plot_height=700)


    explorer.add_tools(hover)
    explorer.add_tools(TapTool(callback=CustomJS(code=PLAY_FROM_EXPLORER,
                                                 args=dict(source=source, path=explorer_audio_path))))

    renderer = explorer.circle(source=source, x='X', y='Y', color='Colour', fill_alpha=0.6, size=15,
                               hover_fill_color='yellow', hover_alpha=1, nonselection_alpha=0.6)

    if explorer_type == 'Customised':
        point_drag = PointDrawTool(renderers=[renderer], add=False)
        explorer.add_tools(point_drag)
        explorer.on_event(PanEnd, pan_python_callback)
        go_back_button = Button(label='Undo Customize')
        go_back_button.on_click(go_back)
        return row(column(explorer, go_back_button), audio_selector)

    elif explorer_type == 'Small':
        return row(explorer, audio_selector)
    elif explorer_type == 'Big':
        return explorer

def make_list_panel():

    path = 'Groove-Explorer-2/static/Part 4 MP3 - Seperate Folders/'

    def get_files(index, labels):
        palette_files = os.listdir(path + labels[index] + '/')
        groove_names = [x[:-4] for x in palette_files]
        return groove_names

    def make_audio_panel():
        PLAY_TEST_AUDIO = """
        var path = 'Groove-Explorer-2/static/Test Audio/List Interface/'
        var index = selector.active;
        var labels = ['A', 'B', 'C', 'D', 'E'];
        var filename = path + labels[index] + '.mp3';
        audio_player.stop_audio();
        audio_player.make_audio(filename);
        audio_player.play_audio();
        """

        STOPCODE = """
        audio_player.stop_audio();
        """

        labels = ['A', 'B', 'C', 'D', 'E']
        audio_selector = RadioGroup(labels=labels, height_policy="auto", sizing_mode='scale_width', active=0)

        play_button = Button(label='Play')
        play_button.js_on_click(CustomJS(args=dict(selector=audio_selector), code=PLAY_TEST_AUDIO))
        stop_button = Button(label='Stop')
        stop_button.js_on_click(CustomJS(code=STOPCODE))
        audio_panel = column(audio_selector, play_button, stop_button)
        return audio_panel

    PLAY_LIST_AUDIO = """
    var groove = cb_obj.labels[cb_obj.active];
    var palette = palette_select.labels[palette_select.active];
    var path = 'Groove-Explorer-2/static/Part 4 MP3 - Seperate Folders/'
    var filename = path + palette + '/' + groove + '.mp3';

    audio_player.stop_audio();
    audio_player.make_audio(filename);
    audio_player.play_audio();
    """

    palette_labels = os.listdir(path)

    opts = {
        0: get_files(0, palette_labels),
        1: get_files(1, palette_labels),
        2: get_files(2, palette_labels),
        3: get_files(3, palette_labels),
        4: get_files(4, palette_labels),
        5: get_files(5, palette_labels),
        6: get_files(6, palette_labels),
        7: get_files(7, palette_labels),
        8: get_files(8, palette_labels),
        9: get_files(9, palette_labels),
        10: get_files(10, palette_labels),
        11: get_files(11, palette_labels),
    }
    palette_file_select = RadioGroup(labels=palette_labels, active=0, sizing_mode='scale_width')
    groove_file_select = RadioGroup(labels=opts[0], height_policy="auto", sizing_mode='scale_width')
    #groove_file_select.js_on_change('active', groove_selection_handler)

    groove_file_select.js_on_change('active', CustomJS(code=PLAY_LIST_AUDIO, args=dict(palette_select=palette_file_select)))
    palette_file_select.js_on_change('active', CustomJS(args=dict(ms=groove_file_select), code="""
    const opts = %s
    ms.labels = opts[cb_obj.active]
""" % opts))

    audio_panel = make_audio_panel()

    return row(palette_file_select, groove_file_select, audio_panel)

list_panel = make_list_panel()
list_tab = Panel(child=list_panel, title="File Browser")

list_panel = make_list_panel()
list_tab = Panel(child=list_panel, title="File Browser")

explorer_part_1 = make_explorer(data_file='Part1_', explorer_type='Small')
small_explorer_tab = Panel(child=explorer_part_1, title="Groove Explorer Part 1 - Small")

explorer_part_2 = make_explorer(data_file='Part3_', explorer_type='Customised')
customised_explorer_tab = Panel(child=explorer_part_2, title="Groove Explorer Part 2 - Customised")

explorer_part_3 = make_explorer(data_file='BIG_', explorer_type='Big')
big_explorer_tab = Panel(child=explorer_part_3, title="Groove Explorer Part 3 - Big")

tabs = Tabs(tabs=[list_tab, small_explorer_tab, customised_explorer_tab, big_explorer_tab])

curdoc().add_root(tabs)