"""
Alumet Energy Visualization Dashboard entry point.
Starts the Dash web application on http://0.0.0.0:8051.
"""

from frontend.app import app
from frontend.layout import create_layout
import frontend.panes  # registers all @app.callback decorators

app.layout = create_layout(app)


def run(*, debug: bool = True, host: str = "0.0.0.0", port: int = 8051) -> None:
    """Start the Dash web dashboard."""
    app.run(debug=debug, host=host, port=port)


if __name__ == "__main__":
    run()
