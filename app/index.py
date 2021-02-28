import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc

from app import app
from apps import point_cloud_app, graph_app,time_series_app

from navbar import Navbar,Footer
from os.path import isfile, join

txt_file = join("../docs/","introduction.md")
with open(txt_file,"r") as f:
    desc = f.read()
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

description = dbc.Card([
        dbc.Container([
            dcc.Markdown(desc,
            style={
                'margin-top': 10,
                'margin-bottom': 10
            })
        ])
    ],
    style={
        'margin-top': 10,
        'margin-bottom': 10
    })
layout = [dbc.Container([
    Navbar(),
    html.Div([
        description,
    ]),


    ]),
    html.Div([
        Footer(),
    ]),
]


@app.callback(Output('page-content', 'children'),
              Input('url', 'pathname'))
def display_page(pathname):
    if pathname == '/point_cloud':
        return point_cloud_app.layout
    elif pathname == '/graph':
        return graph_app.layout
    elif pathname == '/timeseries':
        return time_series_app.layout
    elif pathname == '/':
        return layout
    else:
        return '404'

if __name__ == '__main__':
    app.run_server(debug=True)
