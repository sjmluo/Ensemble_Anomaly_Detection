# -*- coding: utf-8 -*-

# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.
import sys
sys.path.append('..')

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output,ALL,MATCH

import json
import pickle
import re

import dash_cytoscape as cyto
import dash_bootstrap_components as dbc

from app import app
from apps.point_cloud_app import display_description
from navbar import Navbar
from os.path import isfile, join



dataset_dropdown_choices = [
    {
        'label': "Ensemble Graph",
        'value': 'ensemble_graph'
    },

    {
        'label': "Twitter WorldCup 2014",
        'value': 'twitter_worldcup'
    },
    #{
    #    'label': "Yelp",
    #    'value': 'yelpchi'
    #},
]

ensemble_model_dropdown = [
    {
        'label': 'Oddball',
        'value': 'oddball'
    },
    {
        'label': 'SCAN',
        'value': 'scan'
    },
    {
        'label': 'Eigenspoke',
        'value': 'eigenspoke'
    },
]

twitter_model_dropdown = [
    {
        'label': 'Nonnegative Matrix Factorisation (NMF)',
        'value': 'nmf'
    },
]
controls = dbc.Card(
    [
        dbc.FormGroup(
            [
                dbc.Label("Dataset"),
                dcc.Dropdown(
                    id="graph_dataset",
                    options=dataset_dropdown_choices,
                    value=dataset_dropdown_choices[0]['value']
                ),
            ]
        ),
        dbc.FormGroup(
            [
                dbc.Label("Model"),
                dcc.Dropdown(
                    id="graph-model",
                    options=ensemble_model_dropdown,
                    value=ensemble_model_dropdown[0]['value'],
                ),
            ]
        ),
        html.P(
            "This may taken a few seconds to update.",
            className="card-text",
            style={
                'margin-bottom': 10
            })
    ],
    body=True,
)
default_graph_stylesheet = [
    {
        'selector': 'node',
        'style': {
            'label': "data(id)",
            'width': "30%",
            'height': "30%",
        }
    },

]
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
        html.Div(id="graphs",
        style={
            'padding-top': 10
        })

    ]),
])
    #cyto.Cytoscape(
    #    id='cytoscape-two-nodes',
    #    layout={'name': 'circle'},
    #    style={'width': '100%', 'height': '1000px'},
    #    stylesheet=[
    #        {
    #            'selector': 'node',
    #            'style': {
    #                'label': "data(id)",
    #                'width': "10%",
    #                'height': "10%"
    #            }
    #        },
    #        {
    #            'selector': '.red',
    #            'style': {
    #                'background-color': 'red',
    #                'line-color': 'red'
    #            }
    #        },
#
#            {
#                'selector': 'edge',
#                'style': {
#                    'width': 'mapData(weight, 0,50, 1, 10)',
#                    'label': "data(weight)",
#                }
#            },
#
#            {
#                'selector': '[anomaly = 1]',
#                'style': {
#                    'background-color': 'red',
#                    'line-color': 'red'
#                }
#            },
#        ],
#        #elements=elements
#    )



"""
@app.callback(
    Output('graphs', 'elements'),
    Input('graph_dataset', 'value'),)
def graph_dataset_description(graph_dataset):
    if graph_dataset == 'twitter_worldcup':
        pickle_file = 'twitter_worldcup_elements.pickle'
    elif graph_dataset == 'yelpchi':
        pickle_file = 'yelpchi.pickle'

    with open(f"../datasets/{pickle_file}", "rb") as input_file:
        elements = pickle.load(input_file)
    return elements
"""
@app.callback(
    Output('graphs', 'children'),
    Input('graph_dataset', 'value'),)
def create_network(graph_dataset):
    if graph_dataset == 'ensemble_graph':
        graphs = plot_ensemble_network()
    elif graph_dataset == 'twitter_worldcup':
        graphs = plot_twitter_network()
    return graphs

@app.callback(
    Output('graph-model', 'value'),
    Output('graph-model', 'options'),
    Input('graph_dataset', 'value'),)
def update_model_dropdown(graph_dataset):
    if graph_dataset == 'ensemble_graph':
        dropdown = ensemble_model_dropdown
        value = ensemble_model_dropdown[0]['value']
    elif graph_dataset == 'twitter_worldcup':
        dropdown = twitter_model_dropdown
        value = twitter_model_dropdown[0]['value']
    return value,dropdown

def plot_twitter_network():
    with open(f"../datasets/twitter_worldcup_elements.pickle", "rb") as input_file:
        elements = pickle.load(input_file)
    graphs = cyto.Cytoscape(
        id='cytoscape-two-nodes',
        layout={'name': 'circle'},
        style={'width': '100%', 'height': '1000px','border-style':'solid','border-width': '1px'},
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

           {
                'selector': '[anomaly = 1]',
                'style': {
                    'background-color': 'red',
                    'line-color': 'red'
               }
            },
        ],
        elements=elements
    )
    return graphs

def plot_ensemble_network():
    array = []
    for i in range(9):
        with open(f"../datasets/graph/example{i}.pickle", "rb") as input_file:
            elements = pickle.load(input_file)

            network = cyto.Cytoscape(
                id={
                    'type': 'cyto-graph',
                    'index': i
                },
                layout={'name': 'cose'},
                style={'width': '100%', 'height': '250px','border-style':'solid','border-width': '1px'},
                stylesheet=default_graph_stylesheet,
                elements=elements,
                maxZoom=1,
                minZoom=0.5,
                userPanningEnabled=False
            )
        array.append(dbc.Col(network,md=4))
    with open(f"../datasets/graph/barycentre_example.pickle", "rb") as input_file:
        elements = pickle.load(input_file)

        bary_network = cyto.Cytoscape(
            id={
            'type': 'cyto-graph',
            'index': 420
            },
            layout={'name': 'cose'},
            style={'width': '100%', 'height': '500px','border-style':'solid','border-width': '1px'},
            stylesheet=default_graph_stylesheet,
            elements=elements,
            maxZoom=1,
            minZoom=0.5,
            userPanningEnabled=False
        )
    graphs = [dbc.Row(array),dbc.Row(dbc.Col(bary_network,md=12))]

    return graphs

@app.callback(
    Output('graph-data-descript-heading', 'children'),
    Output('graph-data-descript-body', 'children'),
    Input('graph_dataset', 'value'),)
def graph_dataset_description(*args,**kwargs):
    return display_description(*args,**kwargs)

@app.callback(
    Output({'type': 'cyto-graph', 'index': MATCH}, 'stylesheet'),
    Input('graph-model', 'value'),)
def show_labels(graph_model):
    new_styles = [

         {
            'selector': f'[{graph_model}_pred = 1]',
            'style': {
                'background-color': 'red'
            }
        }
    ]
    return default_graph_stylesheet + new_styles
