import dash
import dash_bootstrap_components as dbc
from pathlib import Path

ASSETS_FOLDER = str(Path(__file__).resolve().parent / "assets")
BOOTSTRAP_ICONS = "https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css"

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP, BOOTSTRAP_ICONS],
    assets_folder=ASSETS_FOLDER,
)
app.config.suppress_callback_exceptions = True
