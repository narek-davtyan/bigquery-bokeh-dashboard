import pandas as pd
import numpy as np


# Import multiprocessing libraries
from pandarallel import pandarallel

# Initialization
pandarallel.initialize()

# Load data
min_list_of_columns_to_load = ['company', 'service', 'recommendation', 'easiness', 'rec_sc', 'eas_sc']
df_orig = pd.read_excel(r'CDD1.xlsx', names=min_list_of_columns_to_load)#.astype({'country':'category', 'company':'int16', 'service':'category', 'recommendation':'int8', 'question_one':'string', 'easiness':'int8', 'question_two':'string'})

# Initial data transformation
df_orig['service'] = df_orig['service'].parallel_map(str)

# Create dictionary of all plots, filter lock, filters, data sources
general_dict = {}


def calculate_barycenter(df_temp, country_list):   
    # Create visual data points
    df_tempo = df_temp[['recommendation', 'easiness', 'company']].groupby(['recommendation', 'easiness'],as_index=False).count().rename(columns={'company' : 'sum'}).astype({'sum': 'float32'})
    df_tempy = pd.merge(df_temp, df_tempo, how='left', on=['recommendation', 'easiness'])

    # Calculate size of circles
    df_tempy.loc[~df_tempy['service'].isin(country_list), 'sum'] = 0.0
    df_tempy.loc[df_tempy['sum'] > 25.0, 'sum'] = 25.0
    df_tempy.eval('visual_sum = sum * 2.95', inplace=True)

    # Create visual barycenter with edges
    if len(df_temp) == 0 or len(country_list) == 0:
        barycenter = np.array([0.0, 0.0])
    else:
        barycenter = df_temp[['recommendation', 'easiness']].astype({'recommendation':'float32', 'easiness':'float32'}).mean().to_numpy()
    
    # Create barycenter dataframe
    bary_numpy = df_temp[['recommendation', 'easiness']].astype({'recommendation':'float32', 'easiness':'float32'}).to_numpy()

    row_bary = [barycenter[0], barycenter[1]]
    row_empty = np.empty((1,bary_numpy.shape[1]))
    row_empty.fill(np.nan)

    bary_numpy = np.insert(bary_numpy, range(1, len(bary_numpy)+1, 1), row_bary, axis=0)
    bary_numpy = np.insert(bary_numpy, range(2, len(bary_numpy), 2), row_empty, axis=0)
    bary_data = pd.DataFrame(bary_numpy, columns=['recommendation', 'easiness'])

    return df_tempy, barycenter, bary_data

# Unset initial filter lock
general_dict['filter_called'] = False
# Set initial filters to all
general_dict['filter_list'] = df_orig.service.unique()
general_dict['full_filter_list'] = df_orig.service.unique()

# Calculating filtered dataframe
filtered_df = df_orig.loc[df_orig['service'].isin(general_dict['filter_list'])]
# Calculating new data points, barycenter and its edges
df_points, barycenter, df_bary = calculate_barycenter(filtered_df[min_list_of_columns_to_load], general_dict['filter_list'])

###################################################################################
###################################################################################

from bokeh.models import ColumnDataSource, Callback, Toggle, BoxAnnotation, LabelSet, Label, HoverTool, DataTable, TableColumn, Image, TapTool, Tap, HBar, Plot
from bokeh.plotting import figure, curdoc
from bokeh.layouts import column, row, Spacer

###################################################################################
############################## Visual 3 - Data Table ##############################
# Create data table structure
data_columns = [
        TableColumn(field="company", title="Company"),
        TableColumn(field="service", title="Service"),
    ]
data_source = ColumnDataSource(pd.DataFrame(columns=['service', 'company']))
data_table = DataTable(source=data_source, columns=data_columns, width=400, height=550)

###################################################################################
###################################################################################


###################################################################################
############################## Visual 1 - Points Plot #############################


#---------------------------------------------------------------------------------#
#------------------------------- Static Background -------------------------------#
# Create points plot
general_dict['points_plot'] = figure(x_range=(0, 10), y_range=(0, 10), plot_width=600, plot_height=600, match_aspect=True, tools=['tap'])

# Hide real axis
general_dict['points_plot'].axis.visible = False

# Hide real grid
general_dict['points_plot'].xgrid.grid_line_color = None
general_dict['points_plot'].ygrid.grid_line_color = None

# Define grid lines
general_dict['points_plot'].xaxis.ticker = list(range(11))
general_dict['points_plot'].yaxis.ticker = list(range(11))

# Create color zones
general_dict['points_plot'].circle(x=7.0, y=7.0, radius=1, fill_alpha=1, fill_color='#fbe5d6', radius_units='data', line_color=None, level='underlay')
ba1 = BoxAnnotation(bottom=7, top=10, left=0, right=7, fill_alpha=1, fill_color='#fbe5d6', level='underlay')
ba2 = BoxAnnotation(bottom=0, top=7, left=7, right=10, fill_alpha=1, fill_color='#fbe5d6', level='underlay')
ba3 = BoxAnnotation(bottom=0, top=7, left=0, right=7, fill_alpha=0.3, fill_color='#bf0603', level='underlay')
ba4 = BoxAnnotation(bottom=7, top=10, left=7, right=10, fill_alpha=0.3, fill_color='#538d22', level='underlay')
general_dict['points_plot'].add_layout(ba1)
general_dict['points_plot'].add_layout(ba2)
general_dict['points_plot'].add_layout(ba3)
general_dict['points_plot'].add_layout(ba4)

# Create fake axis lines with ticks and labels
general_dict['points_plot'].line(x=[0, 10], y=[7, 7], line_color='skyblue', level='underlay')
general_dict['points_plot'].line(x=[7, 7], y=[0, 10], line_color='forestgreen', level='underlay')
general_dict['points_plot'].segment(x0=list(range(11)), y0=list(np.array(range(7,8))-0.1)*11,
             x1=list(range(11)), y1=list(np.array(range(7,8))+0.1)*11,
             color='skyblue', line_width=2, level='underlay')
general_dict['points_plot'].segment(x0=list(np.array(range(7,8))-0.1)*11, y0=list(range(11)),
             x1=list(np.array(range(7,8))+0.1)*11, y1=list(range(11)),
             color='forestgreen', line_width=1, level='underlay')
source = ColumnDataSource(data=dict(height=list(range(11)),
                                    weight=list(np.array(range(7,8)))*11,
                                    names=list(range(11))))
labels = LabelSet(x='weight', y='height', text='names', level='glyph',
              x_offset=8, y_offset=2, source=source, render_mode='canvas')
general_dict['points_plot'].add_layout(labels)
labels = LabelSet(x='height', y='weight', text='names', level='glyph',
              x_offset=5, y_offset=-20, source=source, render_mode='canvas')
general_dict['points_plot'].add_layout(labels)

# Create quadrant labels
citation = Label(x=8, y=8, text='Love', render_mode='css')
general_dict['points_plot'].add_layout(citation)
citation = Label(x=3, y=8, text='Frustration', render_mode='css')
general_dict['points_plot'].add_layout(citation)
citation = Label(x=3, y=3, text='Repulsion', render_mode='css')
general_dict['points_plot'].add_layout(citation)
citation = Label(x=8, y=3, text='Frustration', render_mode='css')
general_dict['points_plot'].add_layout(citation)
#----------------------------- ^ Static Background ^ -----------------------------#
#---------------------------------------------------------------------------------#


#---------------------------------------------------------------------------------#
#------------------------------ Ineractive Triggers ------------------------------#

# Filter countries on button click
def callback_h(selected_state):
    # Ignore any individual beahaviour of buttons after 'Select All/None' was triggered
    if general_dict['filter_called']:
        return None

    # Get selected filters from toggle buttons
    selected_country_list = []
    if filter_button1.active:
        general_dict['filter_called'] = True
        for button in buttons:
            button.active = False
        general_dict['filter_called'] = False
        filter_button1.active = False
    elif filter_button3.active:
        general_dict['filter_called'] = True
        for button in buttons:
            button.active = True
            selected_country_list.append(button.name)
        general_dict['filter_called'] = False
        filter_button3.active = False
        if len(selected_country_list) == len(general_dict['full_filter_list']):
            return None
    else:
        for button in buttons:
            if button.active:
                selected_country_list.append(button.name)

    # Setting new filters
    general_dict['filter_list'] = selected_country_list
    # Calculating new filtered dataframe
    filtered_df = df_orig.loc[df_orig['service'].isin(general_dict['filter_list'])]
    # Calculating new data points, barycenter and its edges
    df_points, barycenter, df_bary = calculate_barycenter(filtered_df[min_list_of_columns_to_load], general_dict['filter_list'])
    
    # Create data source for points plot
    general_dict['data_points'] = ColumnDataSource(df_points)
    
    # Attach circle tap callback to new circles
    general_dict['data_points'].selected.on_change('indices', callback)
    
    # Remove old data points
    general_dict['points_plot'].renderers.remove(general_dict['points_plot'].select(name='data_points')[0])
    
    # Plot new data points
    general_dict['points_plot'].circle('recommendation', 'easiness', name='data_points', size='visual_sum', source=general_dict['data_points'], selection_fill_alpha=0.2, selection_color="firebrick", line_width=1, nonselection_line_color="firebrick")
    
    # Remove old barycenter and connecting edges
    if len(general_dict['points_plot'].select(name='bary')) > 0 and len(general_dict['points_plot'].select(name='barypoint')) > 0:
        general_dict['points_plot'].renderers.remove(general_dict['points_plot'].select(name='bary')[0])
        general_dict['points_plot'].renderers.remove(general_dict['points_plot'].select(name='barypoint')[0])

    # Plot new barycenter and connecting edges
    general_dict['points_plot'].line(x='recommendation', y='easiness', source=ColumnDataSource(df_bary), name='bary', line_width=1, level='overlay', color='#2a679d')
    general_dict['points_plot'].circle(x=barycenter[0], y=barycenter[1], color='firebrick', size=barycenter[0]+barycenter[1]+1, name='barypoint', level='overlay')

    # Calculate new scores
    df_emotions = filtered_df[['rec_sc','eas_sc']]

    if len(df_emotions) > 0:
        rec_score = df_emotions['rec_sc'].mean() * 100
        easy_score = df_emotions['eas_sc'].mean() * 100
    else:
        rec_score = 0.0
        easy_score = 0.0
    
    # Update scores
    general_dict['emotions_rec_score'].patch({ 'right' : [(0,rec_score)], 'left' : [(0,rec_score)] })
    general_dict['emotions_easy_score'].patch({ 'right' : [(0,easy_score)], 'left' : [(0,easy_score)] })

# Update data table on circle tap
def callback(attr, old, new):
    recommendations, easinesses = ([],[])

    inds = general_dict['data_points'].selected.indices
    if (len(inds) == 0):
        pass

    for i in range(0, len(inds)):
        recommendations.append(general_dict['data_points'].data['recommendation'][inds[i]])
        easinesses.append(general_dict['data_points'].data['easiness'][inds[i]])
    
    current = df_points.loc[(df_points['recommendation'].isin(recommendations)) & (df_points['easiness'].isin(easinesses)) & (df_points['service'].isin(general_dict['filter_list']))]
    
    data_source.data = {
            'service' : current.service,
            'company' : current.company,
        }
    
#---------------------------- ^ Ineractive Triggers ^ ----------------------------#
#---------------------------------------------------------------------------------#

# Create data source for points plot
general_dict['data_points'] = ColumnDataSource(df_points)

# Attach circle tap callback to circles
general_dict['data_points'].selected.on_change('indices', callback)

# Plot data circles
general_dict['points_plot'].circle('recommendation', 'easiness', name='data_points', size='visual_sum', source=general_dict['data_points'], selection_fill_alpha=0.2, selection_color="firebrick", line_width=1, nonselection_line_color="firebrick")
# general_dict['points_plot'].circle('recommendation', 'easiness', name='data_points', size='visual_sum', alpha=0.4, source=general_dict['data_points'], selection_color="firebrick", selection_alpha=0.4, tags=['country','service'], line_width=1, nonselection_fill_alpha=0.2, nonselection_fill_color="blue", nonselection_line_color="firebrick", nonselection_line_alpha=1.0)

# Plot barycenter and connecting edges
general_dict['bary_points'] = ColumnDataSource(df_bary)
general_dict['points_plot'].line(x='recommendation', y='easiness', source=general_dict['bary_points'], name='bary', line_width=1, level='overlay', color='#2a679d')
general_dict['points_plot'].circle(x=barycenter[0], y=barycenter[1], color='firebrick', size=barycenter[0]+barycenter[1], name='barypoint', level='overlay')

###################################################################################
###################################################################################

###################################################################################
############################ Visual 2 - Buttons Columns ###########################
buttons = []
for country in df_orig.service.unique():
    # Plot buttons
    button = Toggle(label=country, button_type="primary", name=country, width=290)
    button.active = True
    button.on_click(callback_h)
    buttons.append(button)

filter_button1 = Toggle(label='Select None', button_type="default", name='filter1', width_policy='fixed', width=290)
filter_button3 = Toggle(label='Select All', button_type="default", name='filter3', width_policy='fixed', width=290)
filter_button1.active = False
filter_button3.active = False
filter_button1.on_click(callback_h)
filter_button3.on_click(callback_h)
###################################################################################
###################################################################################


###################################################################################
############################# Visual 6 - Emotions Plot ############################

df_emotions = filtered_df[['rec_sc','eas_sc']]

rec_score = df_emotions['rec_sc'].mean() * 100
easy_score = df_emotions['eas_sc'].mean() * 100

general_dict['emotions_rec_score'] = ColumnDataSource(dict(right=[rec_score], left=[rec_score],))
general_dict['emotions_easy_score'] = ColumnDataSource(dict(right=[easy_score], left=[easy_score],))

general_dict['emotions_plot'] = Plot(
    title=None, plot_width=600, plot_height=180, align='center',
    min_border=0, toolbar_location=None, outline_line_color=None, output_backend="webgl")

general_dict['emotions_plot'].add_glyph(HBar(y=0.4, right=0, left=-100, height=0.2, fill_color="#931a25", line_width=0))
general_dict['emotions_plot'].add_glyph(HBar(y=0.0, right=0, left=-100, height=0.2, fill_color="#931a25", line_width=0))
general_dict['emotions_plot'].add_glyph(HBar(y=0.4, right=30, left=0, height=0.2, fill_color="#ffc93c", line_width=0))
general_dict['emotions_plot'].add_glyph(HBar(y=0.0, right=30, left=0, height=0.2, fill_color="#ffc93c", line_width=0))
general_dict['emotions_plot'].add_glyph(HBar(y=0.4, right=70, left=30, height=0.2, fill_color="#b3de69", line_width=0))
general_dict['emotions_plot'].add_glyph(HBar(y=0.0, right=70, left=30, height=0.2, fill_color="#b3de69", line_width=0))
general_dict['emotions_plot'].add_glyph(HBar(y=0.4, right=100, left=70, height=0.2, fill_color="#158467", line_width=0))
general_dict['emotions_plot'].add_glyph(HBar(y=0.0, right=100, left=70, height=0.2, fill_color="#158467", line_width=0))

general_dict['emotions_plot'].add_glyph(general_dict['emotions_rec_score'], HBar(y=0.4, right='right', left='left', height=0.2, fill_color="#1a1c20", line_width=4), name='rec_s')
general_dict['emotions_plot'].add_glyph(general_dict['emotions_easy_score'], HBar(y=0.0, right='right', left='left', height=0.2, fill_color="#1a1c20", line_width=4), name='easy_s')


# Create labels
citation = Label(x=-24, y=0.55, text='Recommendation', render_mode='css', text_color="#4c4c4c", text_font_style='bold')
general_dict['emotions_plot'].add_layout(citation)
citation = Label(x=-12, y=0.16, text='Easiness', render_mode='css', text_color="#4c4c4c", text_font_style='bold')
general_dict['emotions_plot'].add_layout(citation)
citation = Label(x=-82, y=-0.2, text='NEEDS IMPROVEMENT', render_mode='css', text_color="#931a25")
general_dict['emotions_plot'].add_layout(citation)
citation = Label(x=7, y=-0.2, text='GOOD', render_mode='css', text_color="#ffc93c")
general_dict['emotions_plot'].add_layout(citation)
citation = Label(x=40, y=-0.2, text='GREAT', render_mode='css', text_color="#b3de69")
general_dict['emotions_plot'].add_layout(citation)
citation = Label(x=68, y=-0.2, text='EXCELLENT', render_mode='css', text_color="#158467")
general_dict['emotions_plot'].add_layout(citation)
citation = Label(x=-103, y=0.16, text='-100', render_mode='css', text_color="#4c4c4c")
general_dict['emotions_plot'].add_layout(citation)
citation = Label(x=93, y=0.16, text='100', render_mode='css', text_color="#4c4c4c")
general_dict['emotions_plot'].add_layout(citation)
citation = Label(x=1.5, y=0.35, text='0', render_mode='css', text_color="#f4f4f4")
general_dict['emotions_plot'].add_layout(citation)
citation = Label(x=31.5, y=0.35, text='30', render_mode='css', text_color="#f4f4f4")
general_dict['emotions_plot'].add_layout(citation)
citation = Label(x=71.5, y=0.35, text='70', render_mode='css', text_color="#f4f4f4")
general_dict['emotions_plot'].add_layout(citation)
citation = Label(x=1.5, y=-0.05, text='0', render_mode='css', text_color="#f4f4f4")
general_dict['emotions_plot'].add_layout(citation)
citation = Label(x=31.5, y=-0.05, text='30', render_mode='css', text_color="#f4f4f4")
general_dict['emotions_plot'].add_layout(citation)
citation = Label(x=71.5, y=-0.05, text='70', render_mode='css', text_color="#f4f4f4")
general_dict['emotions_plot'].add_layout(citation)

###################################################################################
###################################################################################

# Connect all plots into one object and set layout
curdoc().add_root(row(general_dict['points_plot'], column(row(column(filter_button1, filter_button3, column(buttons)), data_table), Spacer(height=50), general_dict['emotions_plot'])))
