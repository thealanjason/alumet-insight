# Alumet Insight

[![Tests](https://img.shields.io/github/actions/workflow/status/thealanjason/alumet-insight/ci.yml?label=tests)](https://github.com/thealanjason/alumet-insight/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-how--to--use-blue)](docs/how-to-use.md)

Alumet Insight explores and visualizes output from [Alumet-agent](https://alumet-dev.github.io/user-book/start/install.html) measurements. Use the interactive dashboard or command-line tools to summarize experiments, export processed data, and inspect energy, power, utilization, and other metrics over time.

<img src="https://raw.githubusercontent.com/thealanjason/alumet-insight/main/images/layout.png" width="800" alt="Alumet Insight dashboard layout">

## Prerequisites

1. Python >= 3.10. Install via [pip](https://pip.pypa.io/) or [uv](https://github.com/astral-sh/uv).

2. Input configuration and output files of [Alumet-agent](https://alumet-dev.github.io/user-book/start/install.html) measurement. Examples: [energy_measurement experiments](https://github.com/thealanjason/energy_measurement/tree/main/measurement_tools/alumet/experiments)

## Installation

### From PyPI (when published)

```bash
pip install alumet-insight
# or: uv pip install alumet-insight
# or: poetry add alumet-insight
```

### From a clone (development)

```bash
git clone https://github.com/thealanjason/alumet-insight.git
cd alumet-insight

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

# or, with uv (uses uv.lock):
# uv sync --extra dev
```

## Usage

Alumet Insight supports two modes. The **Dashboard** is the primary interface for interactively exploring experiments in the browser. The **Command Line Interface** run summaries, CSV exports, and plot saves from the terminal.

> [!NOTE]
> After `pip install` (or editable install), the `alumet-insight` console script is available. Running `python alumet_insight.py …` from a clone still works.

### Dashboard 

#### Get Started

```bash
alumet-insight dashboard
```

Open `http://localhost:8051` in your browser.

#### How to use the dashboard?

See detailed documentation of how to interactive with the dashboard [here](docs/how-to-use.md).

### Command Line Interface

1. Quick summary:
```bash
alumet-insight cli /path/to/alumet/experiment/dir --summary
```

2. Data processing and export as CSV:
```bash
alumet-insight cli /path/to/alumet/experiment/dir --export-csv /path/to/saved/results
```

with optional `--process-specific` flag to focus on process active region.

3. Visualize the processed data and save as figures:
```bash
alumet-insight cli /path/to/alumet/experiment/dir --export-figures /path/to/saved/results
```

with optional `--process-specific` flag to focus on process active region.
