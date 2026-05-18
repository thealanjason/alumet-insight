import dash
import dash_bootstrap_components as dbc

BOOTSTRAP_ICONS = "https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css"

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP, BOOTSTRAP_ICONS],
    assets_folder="../assets",
)
app.config.suppress_callback_exceptions = True
