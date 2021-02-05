import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc

from app import app
from apps import point_cloud_app, graph_app

from navbar import Navbar,Footer


app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

description = dbc.Card(
    [
        dbc.Container([
            html.H4(
                "Some Title",
                className="card-title",
                style={
                    'margin-top': 10
                }),
            html.P(
                "Some description of the work",
                className="card-text",
                style={
                    'margin-bottom': 10
                }),
    ])
    ]
)

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
    elif pathname == '/':
        return layout
    else:
        return '404'

if __name__ == '__main__':
    app.run_server(debug=True)
