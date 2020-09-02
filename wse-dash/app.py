import dash
import dash_bootstrap_components as dbc

# https://community.plotly.com/t/dash-v1-12-0-release-pattern-matching-callbacks-fixes-shape-drawing-new-datatable-conditional-formatting-options-prevent-initial-call-and-more/38867
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.UNITED] , prevent_initial_callbacks=True)
app.config.suppress_callback_exceptions = True
server = app.server

# more about logic: https://dash.plotly.com/urls
