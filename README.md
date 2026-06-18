# Alumet EDA

Alumet EDA explores and visualizes output from [Alumet-agent](https://alumet-dev.github.io/user-book/start/install.html) measurements. Use the interactive dashboard or command-line tools to summarize experiments, export processed data, and inspect energy, power, utilization, and other metrics over time.

## Prerequisites

1. Pre-install [conda](https://www.anaconda.com/docs/getting-started/miniconda/install/overview) or [micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html)

2. Input configuration and output files of [Alumet-agent](https://alumet-dev.github.io/user-book/start/install.html) measurement. (Examples are shown in `measurement_tools/alumet/experiments/` folder in [energy_measurement](https://github.com/thealanjason/energy_measurement))

## Installation

1. Clone this repository
```bash
git clone https://github.com/thealanjason/alumet-viz.git
```

2. Create and activate the conda environment
```bash
conda env create -f environment.yml
conda activate alumet-viz
```

## Usage

Alumet EDA supports two modes. The **Dashboard** is the primary interface for interactively exploring experiments in the browser. The **Command Line Interface** run summaries, CSV exports, and plot saves from the terminal.

### Dashboard 

#### Get Started

```bash
python alumet_dashboard.py
```

Open `http://localhost:8051` in your browser.

#### How to use the dashboard?

See detailed documentation of how to interactive with the dashboard [here](docs/how-to-use.md).

### Command Line Interface

1. Quick summary:
```bash
python alumet_eda.py /path/to/alumet/experiment/dir --summary
```

2. Data processing and export as CSV:
```bash
python alumet_eda.py /path/to/alumet/experiment/dir --export-csv /path/to/saved/results
```

with optional `--process-specific` flag to focus on process active region.


3. Visualize the processed data and save as figures:
```bash
python alumet_eda.py /path/to/alumet/experiment/dir --export-figure /path/to/saved/results
```

with optional `--process-specific` flag to focus on process active region.

