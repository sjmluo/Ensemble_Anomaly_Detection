# -*- coding: utf-8 -*-

# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.
import sys
sys.path.append('..')
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import plotly.express as px
from dash.dependencies import Input, Output
import plotly.graph_objects as go


import dash_bootstrap_components as dbc

from os import listdir
from os.path import isfile, join
import re
from pyod.models.iforest import IForest
from pyod.models.lof import LOF
from pyod.models.knn import KNN
from pyod.models.pca import PCA
from pyod.models.ocsvm import OCSVM
from pyod.utils.utility import standardizer

from scipy.io import loadmat
from modules.evaluation import EvaluationFramework
from modules.metrics import metrics
from sklearn.model_selection import train_test_split

import json
import pandas as pd
import numpy as np

from flask_caching.backends import FileSystemCache
from dash_extensions.callback import CallbackCache
from app import app
from navbar import Navbar,Footer


dataset_folder = "../datasets/"
onlyfiles = [join(dataset_folder, f) for f in listdir(dataset_folder) if isfile(join(dataset_folder, f))]
data_dict = {}

dataset_dropdown_choices = []
for file in onlyfiles[:11]:
    name = re.search("\/datasets\/(.*?)\.mat",file).group(1)
    data_dict[name] = file
    choice = {
        'label':name, 'value':name
    }
    dataset_dropdown_choices.append(choice)
methods = {
    'iforest': IForest(),
    'knn': KNN(),
    'lof': LOF(),
    'pca': PCA(),
    'ocsvm': OCSVM()
}

model_dropdown = []
for k in methods.keys():
    model_dropdown.append(
        {
        'label':k.upper(),
        'value':k
        }
    )

result_columns = ["Model"] + list(metrics.keys())

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
                dbc.Label("Model"),
                dcc.Dropdown(
                    id="model",
                    options=model_dropdown,
                    value=model_dropdown[0]['value'],
                    multi=True
                ),
            ]
        ),
        dbc.FormGroup(
            [
                html.Label('Visualisation'),
                dcc.RadioItems(
                    id="visualisation",
                    options=[
                        {'label': 'PCA', 'value': 'pca'},
                        {'label': 't-SNE', 'value': 'tsne'},
                    ],
                    value='pca',
                    labelStyle={
                        'display': 'block',
                        'padding-right':'2em'
                    }
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
                id="data-descript-heading",
                className="card-title",
                style={
                    'margin-top': 10
                }),
            html.P(
                id="data-descript-body",
                className="card-text",
                style={
                    'margin-bottom': 10
                }),
    ])
    ]
)

performance_description = dbc.Card(
    [
        dbc.Container([
            html.H4(
                "Performance metrics",
                id="performance-heading",
                className="card-title",
                style={
                    'margin-top': 10
                }),
            html.P(
                "The performance of the models is shown using five different scoring methods.",
                id="performance-body",
                className="card-text",
                style={
                    'margin-bottom': 10
                }),
    ])
    ]
)

datatable_description = dbc.Card(
    [
        dbc.Container([
            html.H4(
                "Predictions",
                id="datatable-heading",
                className="card-title",
                style={
                    'margin-top': 10
                }),
            html.P(
                "The true labels are shown along with the predictions by the selected models.",
                className="card-text"),
            html.P(
                "These can be filtered and consequently plotted to analyse the possible important features for description.",
                className="card-text",
            ),
            html.P(
                "Try using filters like =, <=, >= .",
                className="card-text",
            )
    ],
    style={
        'margin-bottom': 10
    })
    ])

layout =  dbc.Container([
    Navbar("/point_cloud"),
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
    dbc.Row([
        dbc.Col(performance_description,md=4),
        dbc.Col(
            dash_table.DataTable(
                id='table',
            ),
            md=8
        ),
    ],
    style={
        'margin-top': 10,
        'margin-bottom': 20,
    }),
    dbc.Row([
        dbc.Col(datatable_description,md=4),
        dbc.Col(
            dash_table.DataTable(
                id='data-table',
                style_table={
                    'margin-bottom': '60px',
                    'height': '300px',
                    'overflowY': 'auto',
                },
                fixed_rows={'headers': True},
                page_action='none',
                filter_action='custom',
                filter_query='',
            ),
            md=8,
        )
    ]),
    dbc.Row(id='output-graphs'
        #dbc.Col(html.Div(id='output-graphs'),md=6),
    ),
    # Hidden div inside the app that stores the intermediate value
    html.Div(id='data', style={'display': 'none'}),
    html.Div(id='data_reduced', style={'display': 'none'}),
    html.Div(id='labels', style={'display': 'none'}),

    dcc.Store(id='filtered-index'),
])



"""
Callbacks
"""
def display_description(dataset):
    txt_file = join("../docs/datasets",f"{dataset}.md")
    with open(txt_file,"r") as f:
        desc = f.read()
    name = dataset.replace("_", " ")
    heading = f"{name.title()} dataset"
    return heading,dcc.Markdown(desc)

@app.callback(
    Output('data-descript-heading', 'children'),
    Output('data-descript-body', 'children'),
    Input('dataset', 'value'),)
def point_cloud_description(*args,**kwargs):
    return display_description(*args,**kwargs)

@app.callback(
    Output('data', 'children'),
    Input('dataset', 'value'),)
def load_data(dataset):
    mat = loadmat(data_dict[dataset])
    X = mat['X']
    y = mat['y'].ravel()

    X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=1,train_size=0.6)
    X_train_norm, X_test_norm = standardizer(X_train, X_test)

    json_data = {
        'x_train':X_train_norm.tolist(),
        'x_test':X_test_norm.tolist(),
        'y_train': y_train.tolist(),
        'y_test':y_test.tolist()
    }
    return json.dumps(json_data)

@app.callback(
    Output('data_reduced', 'children'),
    Input('data', 'children'),
    Input('visualisation', 'value'))
def reduce_data(data,visualisation):
    data = json.loads(data)
    eva = EvaluationFramework(None)
    x_reduced = eva.dim_reduction(data['x_train'], method=visualisation)
    json_data = {
        "x_train_reduced":x_reduced.tolist()
    }
    return json.dumps(json_data)

@app.callback(
    Output('output-graphs', 'children'),
    Input('data_reduced', 'children'),
    Input('labels', 'children'),
    Input('visualisation', 'value'),
    Input('filtered-index', 'data'),)
def graph_data(data_reduced,labels,visualisation,index):

    index = index['index']
    data = json.loads(data_reduced)
    labels = json.loads(labels)

    x_reduced = np.array(data["x_train_reduced"])[index]
    graphs = []
    for k in labels.keys():
        label = np.array(labels[k]['labels'])[index]
        fig = go.Figure()
        if visualisation == "pca":
            fig.add_trace(
                go.Contour(
                    z=labels[k]['z'],
                    x=labels[k]['x_grid'],
                    y=labels[k]['y_grid'],
                    contours_coloring='lines',
                    line_width=1.5,
                    opacity=0.5,
                    colorbar=dict(
                        title='Anomaly Likelihood', # title here
                        titleside='right',
                        titlefont=dict(
                            size=14,
                            family='Arial, sans-serif')
                    )
                )
            )
        fig.add_trace(go.Scatter(
            x=x_reduced[label=="Normal",0],
            y=x_reduced[label=="Normal",1],
            mode='markers',
            marker=dict(
                color='blue',
            ),
            name="Normal"
        ))
        fig.add_trace(go.Scatter(
            x=x_reduced[label=="Anomaly",0],
            y=x_reduced[label=="Anomaly",1],
            mode='markers',
            marker=dict(
                color='red',
            ),
            name="Anomaly"
        ))

        fig.update_layout(
            legend_title="Predicted Class",
            xaxis_title="",
            yaxis_title="",
            legend_x=0.01,
            legend_y=0.99,
            title=k.upper(),
        )

        graphs.append(
            dbc.Col(
            dcc.Graph(
                id=f'graph{k}',
                figure=fig,
            ),md=6
        )
        )
    return graphs

@app.callback(
    Output('labels', 'children'),
    Output('table', 'columns'),
    Output('table', 'data'),
    Input('data', 'children'),
    Input('model', 'value'),
    Input('visualisation', 'value'),
    )
def train_model(data,model,visualisation):
    data = json.loads(data)
    if isinstance(model,str):
        model = [model]
    df = pd.DataFrame(columns=result_columns)

    model_labels = {}
    for m in model:
        eva = EvaluationFramework(methods[m])

        eva.fit(np.array(data['x_train']))
        y_pred_train = eva.predict(np.array(data['x_train']))
        y_pred_test = eva.predict(np.array(data['x_test']))

        scores = eva.score(np.array(data['y_test']),y_pred_test)
        scores['Model'] = m
        new_scores = pd.DataFrame([scores])
        df = df.append(new_scores,ignore_index=True)
        model_results = {}
        if visualisation == "pca":
            x_grid,y_grid,_,z = eva.contour(np.array(data['x_train']))

            model_results = {
                'x_grid':x_grid.tolist(),
                'y_grid': y_grid.tolist(),
                'z': z[:,1].reshape((x_grid.shape[0],x_grid.shape[0])).tolist()
            }
        labels = []
        for ele in y_pred_train:
            if ele == 1:
                labels.append('Anomaly')
            else:
                labels.append('Normal')
        model_results['labels'] = labels
        model_labels[m] = model_results
    columns = [{"name":k,"id":k} for k in df.columns]

    df = df.round(2)
    return json.dumps(model_labels),columns,df.to_dict("records")


@app.callback(
    Output('data-table', 'columns'),
    Input('data', 'children'),
    Input('labels', 'children'),)
def tabulate_data(data,labels):
    data = json.loads(data)
    labels = json.loads(labels)
    x_train = data['x_train']
    df = pd.DataFrame(x_train,
            columns=[f"feature-{i}" for i in range(len(x_train[0]))]).round(2)
    df['y_train'] = ["Anomaly" if k == 1 else "Normal" for k in data["y_train"]]

    columns = []
    for k in range(len(df.columns)):
        if 'y' not in df.columns[k]:
            columns.append({"name": f"Feat. {k}","id":f"feature-{k}",'type':'numeric'})
    columns = columns[:5]
    columns.append({
        "name": "True",
        "id":'y_train',
        'type':'text'
    })
    for ele in labels.keys():
        columns.append({
        "name": f"{ele.upper()} Pred.",
        "id":f"{ele}_pred",
        'type':'text'
        })
    return columns


operators = [['ge ', '>='],
             ['le ', '<='],
             ['lt ', '<'],
             ['gt ', '>'],
             ['ne ', '!='],
             ['eq ', '='],
             ['contains '],
             ['datestartswith ']]


def split_filter_part(filter_part):
    for operator_type in operators:
        for operator in operator_type:
            if operator in filter_part:
                name_part, value_part = filter_part.split(operator, 1)
                name = name_part[name_part.find('{') + 1: name_part.rfind('}')]

                value_part = value_part.strip()
                v0 = value_part[0]
                if (v0 == value_part[-1] and v0 in ("'", '"', '`')):
                    value = value_part[1: -1].replace('\\' + v0, v0)
                else:
                    try:
                        value = float(value_part)
                    except ValueError:
                        value = value_part

                # word operators need spaces after them in the filter string,
                # but we don't want these later
                return name, operator_type[0].strip(), value

    return [None] * 3


@app.callback(
    Output('data-table', "data"),
    Output('filtered-index', "data"),
    Input('data', 'children'),
    Input('data-table', "filter_query"),
    Input('labels', 'children'),
    )
def update_table(data,filter,labels):
    data = json.loads(data)
    labels = json.loads(labels)
    x_train = data['x_train']
    df = pd.DataFrame(x_train,
            columns=[f"feature-{i}" for i in range(len(x_train[0]))]).round(2)
    df['y_train'] = ["Anomaly" if k == 1 else "Normal" for k in data["y_train"]]

    for ele in labels.keys():
        df[f"{ele}_pred"] = labels[ele]['labels']
    filtering_expressions = filter.split(' && ')
    dff = df
    for filter_part in filtering_expressions:
        col_name, operator, filter_value = split_filter_part(filter_part)

        if operator in ('eq', 'ne', 'lt', 'le', 'gt', 'ge'):
            # these operators match pandas series operator method names
            dff = dff.loc[getattr(dff[col_name], operator)(filter_value)]
        elif operator == 'contains':
            dff = dff.loc[dff[col_name].str.contains(filter_value)]
        elif operator == 'datestartswith':
            # this is a simplification of the front-end filtering logic,
            # only works with complete fields in standard format
            dff = dff.loc[dff[col_name].str.startswith(filter_value)]
    index = {
        'index': dff.index.values
    }
    return dff.to_dict('records'),index

if __name__ == '__main__':
    app.run_server(debug=True)
