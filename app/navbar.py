import dash_bootstrap_components as dbc
import dash_html_components as html

def Navbar(current_app=None):
    PLOTLY_LOGO = "https://images.plot.ly/logo/new-branding/plotly-logomark.png"

    navbar = dbc.Navbar(
        [
            html.A(
                # Use row and col to control vertical alignment of logo / brand
                dbc.Row(
                    [
                        dbc.Col(html.Img(src=PLOTLY_LOGO, height="30px")),
                        dbc.Col(dbc.NavbarBrand("SL Group", className="ml-2")),
                    ],
                    align="center",
                    no_gutters=True,
                ),
                href="/",
            ),
            dbc.Nav(
                [
                    dbc.NavItem(dbc.NavLink("Point Cloud", href="/point_cloud",active=current_app=="/point_cloud")),
                    dbc.NavItem(dbc.NavLink("Time Series", href="/timeseries",active=current_app=="/timeseries")),
                    dbc.NavItem(dbc.NavLink("Graph", href="/graph",active=current_app=="/graph")),
                ],
                pills=True,
            ),
        ],
        color="light",
        dark=False,
    )
    return navbar

def Footer():
    footer = html.Footer(
        dbc.Container([
            dbc.Navbar(
                [
                    html.A(
                        # Use row and col to control vertical alignment of logo / brand
                        dbc.Row(
                            [
                                dbc.Col(dbc.NavbarBrand("Authors", className="ml-2")),
                            ],
                            align="center",
                            no_gutters=True,
                        ),
                    ),
                    dbc.Nav(
                        [
                            dbc.Col(dbc.NavItem("Simon Luo"),md=1,),
                            dbc.Col(dbc.NavItem("Lamiae Azizi"),md=1),
                            dbc.Col(dbc.NavItem("Harrison Nguyen",className="mr-2"),md=1),
                            dbc.Col(dbc.NavItem("Prasad Cheema",className="ml-1"),md=1),
                            dbc.Col(dbc.NavItem("Michael Lin"),md=1),
                            dbc.Col(dbc.NavItem("Gerry How"),md=1),
                        ],
                    ),
                    dbc.Nav(
                        [
                            dbc.Col(dbc.NavItem("Contact:"),md=2,align="center"),
                            dbc.Col(dbc.NavItem(dbc.NavLink("s.luo{at}sydney{dot}edu{dot}au",href="mailto:s.luo{at}sydney{dot}edu{dot}au")),md=1,)
                        ]
                    ),
                ],
            )
        ]),
        className="footer bg-light",
        style={
            'position': 'absolute',
            'width': '100%',
        }
    )
    return footer
