# -*- coding: utf-8 -*-

# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.
import sys
sys.path.append('..')

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

import json
import pickle
import re

import dash_cytoscape as cyto
import dash_bootstrap_components as dbc

from app import app
from apps.point_cloud_app import display_description
from navbar import Navbar
from os.path import isfile, join

with open(r"../datasets/twitter_worldcup_elements.pickle", "rb") as input_file:
    elements = pickle.load(input_file)

dataset_dropdown_choices = [
    {
        'label': "Twitter WorldCup 2014",
        'value': 'twitter_worldcup'
    },
]

model_dropdown = [
    {
        'label': "Outlier Link Detection",
        'value': 'link'
    },
    {
        'label': "Outlier Node Detection",
        'value': 'node'
    },
]
controls = dbc.Card(
    [
        dbc.FormGroup(
            [
                dbc.Label("Dataset"),
                dcc.Dropdown(
                    id="dataset",
                    options=dataset_dropdown_choices,
                    value=dataset_dropdown_choices[0]['value']
                ),
            ]
        ),
        dbc.FormGroup(
            [
                dbc.Label("Problem"),
                dcc.Dropdown(
                    id="model",
                    options=model_dropdown,
                    value=model_dropdown[0]['value'],
                ),
            ]
        ),
    ],
    body=True,
)

data_description = dbc.Card(
    [
        dbc.Container([
            html.H4(
                id="graph-data-descript-heading",
                className="card-title",
                style={
                    'margin-top': 10
                }),
            html.P(
                id="graph-data-descript-body",
                className="card-text",
                style={
                    'margin-bottom': 10
                }),
    ])
    ]
)

layout = html.Div([
    dbc.Container([
        Navbar("/graph"),
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
    ]),

    cyto.Cytoscape(
        id='cytoscape-two-nodes',
        layout={'name': 'circle'},
        style={'width': '100%', 'height': '1000px'},
        stylesheet=[
            {
                'selector': 'node',
                'style': {
                    'label': "data(id)",
                    'width': "10%",
                    'height': "10%"
                }
            },
            {
                'selector': '.red',
                'style': {
                    'background-color': 'red',
                    'line-color': 'red'
                }
            },

            {
                'selector': 'edge',
                'style': {
                    'width': 'mapData(weight, 0,50, 1, 10)',
                    'label': "data(weight)",
                }
            },
        ],
        elements=elements
    )
])

@app.callback(
    Output('graph-data-descript-heading', 'children'),
    Output('graph-data-descript-body', 'children'),
    Input('dataset', 'value'),)
def graph_dataset_description(*args,**kwargs):
    return display_description(*args,**kwargs)
