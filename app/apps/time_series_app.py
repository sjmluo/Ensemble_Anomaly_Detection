# -*- coding: utf-8 -*-

# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.
import sys
sys.path.append('..')
import os
import plotly.graph_objects as go

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output,ALL,MATCH

import json
import pickle
import re
import mat4py
import numpy as np
import pandas as pd
import dash_bootstrap_components as dbc
import plotly.express as px

from app import app
from apps.point_cloud_app import display_description
from navbar import Navbar

from modules.timeSeriesModels.TimeSeriesAnalysis import TensorDecomp,MatrixProfile,ChangePoint



dataset_dropdown_choices = [
    {
        'label': "Aircraft",
        'value': 'aircraft'
    },
    {
        'label': "Building",
        'value': 'building'
    },

]

model_dropdown = [
    {
        'label': 'Change Point',
        'value': 'change_point'
    },
    {
        'label': 'Matrix Profile',
        'value': 'matrix_profile'
    },
    {
        'label': 'Tensor Decomposition',
        'value': 'tensor_decomp'
    },
]
controls = dbc.Card(
    [
        dbc.FormGroup(
            [
                dbc.Label("Dataset"),
                dcc.Dropdown(
                    id="timeseries-dataset",
                    options=dataset_dropdown_choices,
                    value=dataset_dropdown_choices[0]['value']
                ),
            ]
        ),
        dbc.FormGroup(
            [
                dbc.Label("Model"),
                dcc.Dropdown(
                    id="timeseries-model",
                    options=model_dropdown,
                    value=model_dropdown[0]['value'],
                ),
            ]
        ),
        dbc.FormGroup(
            [
                dbc.Label("Sensor Selection"),
                dcc.Checklist(
                    value = [1],
                    labelStyle =
                    {'display': 'inline-block'},
                    id='timeseries-sensor'
                )
            ]
        ),
    ],
    body=True,
)

dataset_info = {
    'aircraft': {
        'nsensors':5
    },
    'building': {
        'nsensors':24
    },
}
X_list = []
y_list = []

data_path = os.path.abspath(os.path.join('../data/sensor_data')) + '/'
building_dataset = np.array([
    mat4py.loadmat(data_path + f'Building_Sensor{i}.mat')['X'] for i in range(1,dataset_info['building']['nsensors']+1)
])



aircraft_dataset = np.array([
    mat4py.loadmat(data_path + f'Aircraft_Sensor{i}.mat')['X'] for i in range(1,dataset_info['aircraft']['nsensors']+1)
])

data_description = dbc.Card(
    [
        dbc.Container([
            html.H4(
                id="timeseries-data-descript-heading",
                className="card-title",
                style={
                    'margin-top': 10
                }),
            html.P(
                id="timeseries-data-descript-body",
                className="card-text",
                style={
                    'margin-bottom': 10
                }),
    ])
    ])


layout = html.Div([
    dbc.Container([
        Navbar("/timeseries"),
        html.Div([
            dbc.Row(
                [
                    dbc.Col(controls, md=4),
                    dbc.Col(data_description, md=8),
                ],
                align="top",
                style={
                    'padding-top': 10
                }),
            ]),
        html.Div(id="timeseries-data-plot"),
        html.H4("Results"),
        html.Div(id="timeseries-data-result"),
    ]),
])

@app.callback(
    Output('timeseries-data-descript-heading', 'children'),
    Output('timeseries-data-descript-body', 'children'),
    Input('timeseries-dataset', 'value'),)
def timeseries_dataset_description(*args,**kwargs):
    return display_description(*args,**kwargs)

@app.callback(
    Output('timeseries-sensor', 'options'),
    Input('timeseries-dataset', 'value'),)
def timeseries_sensor_option(timeseries_dataset):
    try:
        nsensors = dataset_info[timeseries_dataset]['nsensors']
    except:
        raise ValueError

    model_dropdown = [{'label': i, 'value': i} for i in range(1,nsensors+1)]

    return model_dropdown

@app.callback(
    Output('timeseries-data-plot', 'children'),
    Input('timeseries-dataset', 'value'),
    Input('timeseries-sensor', 'value'),)
def plot_timeseries_data(timeseries_dataset,timeseries_sensor):
    if isinstance(timeseries_sensor,int):
        timeseries_sensor = [timeseries_sensor]
    fig = plot_dataset(timeseries_dataset,timeseries_sensor)
    return dcc.Graph(figure=fig)


@app.callback(
    Output('timeseries-data-result', 'children'),
    Input('timeseries-dataset', 'value'),
    Input('timeseries-model', 'value'),
    Input('timeseries-sensor', 'value'),)
def plot_timeseries_data(timeseries_dataset,timeseries_model,timeseries_sensor):
    if isinstance(timeseries_sensor,int):
        timeseries_sensor = [timeseries_sensor]
    if timeseries_model == 'tensor_decomp':
        graphs = tensor_decomp(timeseries_dataset,timeseries_sensor)
    elif timeseries_model == 'change_point':
        graphs = change_point(timeseries_dataset,timeseries_sensor)
    elif timeseries_model == 'matrix_profile':
        graphs = matrix_profile(timeseries_dataset,timeseries_sensor)
    return graphs

def matrix_profile(timeseries_dataset, timeseries_sensor):
    MP = MatrixProfile(data_path = data_path)
    MP.fit(
        data_type = timeseries_dataset.title(), sensor_nums = timeseries_sensor)
    graphs = []
    motif_fig = px.line(
        x = MP.motif_listX[0],
        y = MP.motif_listy[0],
        title = "Matrix Profile Significant Motif")
    motif_fig.update_layout(
        xaxis_title="",
        yaxis_title="",
        width=500,
        height=400,
    )
    motif_fig.update_xaxes(showticklabels=False)
    motif_fig.update_yaxes(showticklabels=False)
    graphs.append(
        dcc.Graph(
            figure = motif_fig,))
    matrix_figure = go.Figure()
    semantic_figure = go.Figure()

    for i in range(len(MP.mp_dict)):
        matrix_figure.add_trace(
            go.Scatter(
            x=np.arange(len(MP.mp_list[i])),
            y=MP.mp_list[i],
            mode='lines',
            name=f"Sensor {timeseries_sensor[i]}"
        ))
        semantic_figure.add_trace(
            go.Scatter(
            x=np.arange(len(MP.mp_list[i])),
            y=MP.cac_list[i],
            mode='lines',
            name=f"Sensor {timeseries_sensor[i]}"
        ))
    ## add the barycenter
    matrix_figure.add_trace(
        go.Scatter(
            x=np.arange(len(MP.mp_list[i])),
            y=MP.mp_bary_list[0],
            mode='lines',
            name=f"Barycentre",
            line=dict(
                width = 5,
                color = 'black'
            ),
            ))
    semantic_figure.add_trace(
        go.Scatter(
        x=np.arange(len(MP.mp_list[i])),
        y=MP._moving_average(MP.cac_bary_list[0],100),
        mode='lines',
        name=f"Barycentre",
        line=dict(
            width = 5,
            color = 'black'
        ),
    ))
    semantic_figure.update_layout(
        xaxis=dict(
            autorange=True,
            rangeslider=dict(
                autorange=True,
            ),
    ))
    matrix_figure.update_layout(
        xaxis=dict(
            autorange=True,
            rangeslider=dict(
                autorange=True,
            ),
    ))
    semantic_figure.update_layout(
        title="Semantic Segmenter",
    )
    matrix_figure.update_layout(
        title="Matrix Profiles",
    )
    graphs.append(
        dcc.Graph(figure=matrix_figure)
    )
    graphs.append(
        dcc.Graph(figure=semantic_figure)
    )
    return graphs

def tensor_decomp(timeseries_dataset,timeseries_sensor):
    TD = TensorDecomp(num_dims=2, data_path=data_path)
    TD.fit(
        data_type=timeseries_dataset.title(), sensor_nums=timeseries_sensor)
    y_labels = TD.arr_stacked_y.flatten()
    if timeseries_dataset == 'building':
        nlabels = 5
        legend = ['Healthy','Damage Level 1',
                   'Damage Level 2', 'Damage Level 3',
                   'Damage Level 4']
    elif timeseries_dataset == 'aircraft':
        nlabels = 3
        legend = ['Healthy Take-Off','Healthy Climb',
                    'Damage Climb']
    fig = go.Figure()
    for i in range(nlabels):
        fig.add_trace(
            go.Scatter(
            x=TD.dim1_x[y_labels==i],
            y=TD.dim1_y[y_labels==i],
            mode='markers',
            marker=dict(
                color='blue',
            ),
            name=legend[i]
        ))
    fig.update_layout(
        legend_title="Damage Type",
        xaxis_title="Magnitude",
        yaxis_title="Magnitude",
        legend_x=0.01,
        legend_y=0.99,
        title="Tensor Decomposition",
    )

    return dcc.Graph(figure=fig)

def plot_dataset(timeseries_dataset,timeseries_sensor):
    if timeseries_dataset == "aircraft":
        data = aircraft_dataset.reshape((aircraft_dataset.shape[0],-1)).T
        max_time = data.shape[0]
    elif timeseries_dataset == "building":
        data = building_dataset.reshape((building_dataset.shape[0],-1)).T
        max_time = 13170
    columns = [f"Sensor {ele}" for ele in timeseries_sensor]
    temp = np.array(timeseries_sensor)-1
    df = pd.DataFrame(data[:,temp],columns=columns)
    df["Time"] = np.arange(1,df.shape[0]+1)

    fig = px.line(df[:max_time], x='Time', y=df.columns,
                 title=f"{timeseries_dataset.title()} dataset")
    fig.update_traces(line=dict(width=0.5))
    fig.update_xaxes(rangeslider_visible=True)

    if timeseries_dataset == "aircraft":
        fig.add_vrect(x0=0, x1=2000,
              annotation_text="Take off Phase", annotation_position="top left",line_width=0)
        fig.add_vrect(x0=2000, x1=4000,
              annotation_text="Climb Phase", annotation_position="top left",
              fillcolor="green", opacity=0.1, line_width=0)
        fig.add_vrect(x0=4000, x1=6000,
              annotation_text="Climb Phase (Damaged)", annotation_position="top left",
              fillcolor="red", opacity=0.1, line_width=0)
    if timeseries_dataset == "building":
        break_points = [5170,7170,9170,11170]
        for i,ele in enumerate(break_points):
            fig.add_vrect(x0=ele, x1=ele+2000,
              annotation_text=f"Damage Level {i+1}", annotation_position="top left",line_width=0,
              fillcolor = "red", opacity=0.1+i*0.1)
    fig.update_layout(
        legend_title="Sensor",
        yaxis_title="Magnitude",
    )
    return fig

def change_point(timeseries_dataset,timeseries_sensor):

    CP = ChangePoint(data_path=data_path)
    CP.fit(
        data_type = timeseries_dataset.title(),
        sensor_nums = timeseries_sensor,
        min_size=200)

    chg_pt_locs = CP.predict()

    relevant_chg_pts = CP.chg_pts[CP.chg_pts > 2000][:-1]

    fig = plot_dataset(timeseries_dataset,timeseries_sensor)
    for ele in relevant_chg_pts:
        fig.add_vline(
            ele,
            line_width=5,
            line_dash='dash',
            line_color="blue",name='Change Points')
    fig.update_layout(
        title="Change Point",
    )
    return [dcc.Graph(figure=fig)]
