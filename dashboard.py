"""
Alumet Energy Visualization Dashboard entry point.
Starts the Dash web application on http://0.0.0.0:8051.
"""

from frontend.app import app
from frontend.layout import create_layout
import frontend.panes  # registers all @app.callback decorators

app.layout = create_layout(app)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8051)
