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
                        dbc.Col(dbc.NavbarBrand("Some Brand", className="ml-2")),
                    ],
                    align="center",
                    no_gutters=True,
                ),
                href="/",
            ),
            dbc.Nav(
                [
                    dbc.NavItem(dbc.NavLink("Point Cloud", href="/point_cloud",active=current_app=="/point_cloud")),
                    dbc.NavItem(dbc.NavLink("Time Series", href="/",active=current_app=="/")),
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
                        href="/",
                    ),
                    dbc.Nav(
                        [
                            dbc.NavItem(dbc.NavLink("Name", href="/point_cloud",)),
                            dbc.NavItem(dbc.NavLink("Some other name", href="/",)),
                            dbc.NavItem(dbc.NavLink("Last Guys name", href="/graph",)),
                        ],
                    ),
                ],
            )
        ]),
        className="footer bg-light",
        style={
            'position': 'absolute',
            'bottom': 0,
            'width': '100%',
        }
    )
    return footer
