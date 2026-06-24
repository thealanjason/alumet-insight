# AlumetInsight

AlumetInsight is an open-source Python tool for interactively exploring and visualizing energy, power, and resource-utilization measurements produced by [Alumet-agent](https://alumet-dev.github.io/user-book/start/install.html).
It provides both a browser-based dashboard and a command-line interface for summarizing experiments, exporting processed data, , and inspect energy, power, utilization, and other metrics over time.

<img src="./images/layout.png" width=800>

## Table of Contents

1. [Abstract](#abstract)
2. [Keywords](#keywords)
3. [Overview](#overview)
   - [Introduction](#introduction)
   - [Implementation and Architecture](#implementation-and-architecture)
   - [Usage](#usage)
     - [Tutorial: Your First Experiment Analysis](#tutorial-your-first-experiment-analysis)
     - [Dashboard Reference](#dashboard-reference)
     - [Command-Line Interface Reference](#command-line-interface-reference)
   - [Quality Control](#quality-control)
4. [Availability](#availability)
   - [Operating System](#operating-system)
   - [Programming Language](#programming-language)
   - [Dependencies](#dependencies)
   - [Installation](#installation)
   - [Software Location](#software-location)
   - [Contributors](#contributors)
5. [Reuse Potential](#reuse-potential)
6. [Citation](#citation)
7. [Contributing](#contributing)
8. [Contact](#contact)
9. [License](#license)
10. [Funding and Acknowledgments](#funding-and-acknowledgments)
11. [Competing Interests](#competing-interests)
12. [References](#references)

## Abstract

Measuring the energy consumption and resource utilization of high-performance computing (HPC) workloads is essential for understanding software efficiency and driving sustainable system design. 
[Alumet-agent](https://alumet-dev.github.io/user-book/start/install.html) provides a lightweight, configurable framework for collecting such measurements, producing CSV output files containing timestamped records of energy, power, and hardware utilization
metrics. However, no dedicated tool exists for the interactive exploration and comparative analysis of these outputs across experiments. We present AlumetInsight, an open-source Python package that fills this gap. AlumetInsight offers an interactive browser-based dashboard built on Plotly Dash, allowing researchers to load, filter, and compare multiple Alumet-agent experiment outputs side-by-side in real time. It also exposes a command-line
interface (CLI) for scripted summaries, CSV exports, and figure generation in automated pipelines. AlumetInsight is developed openly on GitHub and is released under the MIT License.

<!-- TODO: Add sentence with Zenodo DOI / repository identifier once archived. -->

## Overview 

### Introduction

Modern scientific computing increasingly relies on understanding the energy
footprint of numerical workloads, both for cost efficiency and for
reproducibility: two runs of the same code on the same hardware may differ
in power draw due to thermal throttling, NUMA effects, or background load.
Capturing and making sense of these dynamics requires tooling that is both
lightweight enough to deploy on production HPC nodes and expressive enough to
surface meaningful trends.

[Alumet-agent](https://alumet-dev.github.io/user-book/start/install.html) addresses the measurement side of this challenge: it is a
Rust-based agent that samples configurable hardware counters (CPU energy via RAPL, GPU power, memory bandwidth, process-level CPU utilization, and others) at sub-second intervals and writes results to CSV files. 
While Alumet-agent's output format is well-defined, no purpose-built analysis layer existed to help researchers move quickly from raw CSV files to insight.
Dedicated HPC monitoring dashboards (e.g., Grafana and InfluxDB stacks) are oriented toward real-time streaming rather than post-hoc exploratory analysis of stored experiment files.
AlumetInsight closes this gap by providing a cohesive, experiment-centric interface that understands Alumet-agent's CSV schema natively, enabling researchers to load one or multiple experiment directories, select metrics and time windows interactively, and export processed data or figures without writing
any additional code.

<!-- TODO: Confirm and expand the list of related tools above.
     Consider adding a comparison table. -->

## Implementation and Architecture

AlumetInsight is implemented in Python and organized around two main modules:

`utils.py`: Core data processing layer.
This module handles discovery and ingestion of Alumet-agent experiment directories. 
It locates CSV files by convention within the experiment directory
structure, parses the Alumet-agent column schema (timestamp, metric name, value, unit, source), and returns tidy pandas DataFrames suitable for downstream
analysis. 
Utility functions for aggregation (mean, max, time-windowed
statistics) and for process-specific filtering are also implemented here.

`dashboard.py`: Interactive visualisation layer.
The dashboard is built on Plotly Dash, a Python framework that renders reactive web applications without requiring a separate
JavaScript front end. 
The layout is divided into a control panel (experiment
directory selector, metric picker, time range slider, process-specific toggle) and a main panel (time-series plots, summary statistics table, and a metric
comparison view). 
Callbacks connect UI controls to re-renders of the figures
in real time.

The overall data flow is:

<!-- TODO: Add an architecture diagram such as mermaid flowchart -->

## Usage

AlumetInsight supports two modes of operation. The Dashboard is the
primary interface for interactive exploration in a browser. The CLI
is designed for scripted, reproducible workflows in pipelines or Makefiles.

Tutorial: Your First Experiment Analysis

This tutorial walks through loading a real Alumet-agent experiment output, exploring metrics in the dashboard, and exporting a summary CSV.

### Prerequisites

You have completed [Installation](#installation) and have an Alumet-agent experiment directory at hand. Example outputs are available at [energy_measurement](https://github.com/thealanjason/energy_measurement) under `measurement_tools/alumet/experiments/`.

#### Step 1. Activate the environment and start the dashboard.

```bash
conda activate alumet-insight
python dashboard.py
```

Open `http://localhost:8051` in your browser.
You will see the AlumetInsight dashboard with an empty experiment panel.

#### Step 2. Load an experiment directory.

In the configuration panel on the left, input the directory path of your Alumet-agent experiment folder. AlumetInsight will discover all CSV files inside and populate the metric selector in the main dashboard.

<!-- TODO: Add a screenshot here (images/tutorial_step2.png). -->

#### Step 3. Exploration.

Select the metric categories from the dropdown. The main panel will render a time-series plot showing power draw over the experiment duration.
For a detailed walkthrough of all dashboard controls, see
docs/how-to-use.md.

#### Step 4. Export the processed summary.

In each plot shown in the dashboard, there is an button for user to click Export CSV to download a tidy summary of the selected metric. Alternatively, use the CLI (see below) to generate the same output non-interactively:

1. Quick summary.
```bash
python alumet_eda.py /path/to/alumet/experiment/dir --summary
```

2. Data processing and export as CSV.
```bash
python alumet_eda.py /path/to/alumet/experiment/dir --export-csv /path/to/saved/results
```

with optional `--process-specific` flag to focus on process active region.


3. Visualize the processed data and save as figures.
```bash
python alumet_eda.py /path/to/alumet/experiment/dir --export-figure /path/to/saved/results
```

with optional `--process-specific` flag to focus on process active region.

### Quality Control

After installation, verify the environment is correctly configured by running the dashboard against the example data in [energy_measurement](https://github.com/thealanjason/energy_measurement):
```bash
conda activate alumet-insight
python dashboard.py
# Navigate to http://localhost:8051 and load the example directory
```

A successful launch without import errors and a rendered dashboard confirm that all dependencies are installed correctly.


## Availability

### Operating System

Linux (tested on Ubuntu 24.04 LTS and later); macOS (tested on macOS 26.5 Tahoe); Windows (not tested, but no known platform-specific dependencies).

### Programming Language

Python 3.11 or later.

### Dependencies

A fully pinned environment is provided in `environment.yml`.

### Installation

>[!NOTE] Prerequisites: [conda](https://www.anaconda.com/docs/getting-started/miniconda/install/overview) or [micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html) must be installed.

```bash
# 1. Clone the repository
git clone https://github.com/thealanjason/alumet-insight.git
cd alumet-insight

# 2. Create and activate the conda environment
conda env create -f environment.yml
conda activate alumet-insight
```

### Software Location

**Code repository**

| Field | Value |
|---|---|
| Name | GitHub |
| Persistent identifier | https://github.com/thealanjason/AlumetInsight |
| Licence | MIT |
| Date published | <!-- TODO: Add date --> |

**Archive (for peer review)**

| Field | Value |
|---|---|
| Name | Zenodo |
| Persistent identifier | <!-- TODO: Add Zenodo DOI after depositing --> |
| Licence | MIT |
| Version | <!-- TODO: Tag a release version --> |
| Date published | <!-- TODO: --> |

**List of Contributors**

- Chia-Hao Chang: Software development, documentation, 
- Alan Correa: Conceptualization, scientific supervision, methodological guidance and critical review of the software design.

See also [Contributing](#contributing) for how to join the project.

## Reuse Potential

AlumetInsight is designed to be reusable beyond its original development context. The core data processing layer (`utils.py`) makes no assumptions about the application domain of the measured workload: it treats Alumet-agent CSV output as a generic time-series of labelled metric samples. Any researcher
using Alumet-agent to profile HPC applications, whether in fluid dynamics, machine learning training, molecular dynamics, or bioinformatics, can use AlumetInsight dircetly for insightful exploration of the energy consumption.

The dashboard is parameterized through the control panel at runtime, making it straightforward to adapt flexibly when Alumet-agent adds new measurement plugins. 
Users wishing to extend the tool can contribute new aggregation strategies in `utils.py` or additional dashboard
panels in `dashboard.py` by following the contribution guidelines.

## Citation

If you use Alumet Insight in your research, please cite the associated JORS software Metapaper:

```bibtex
@article{alumet-insight-jors,
  author  = {<!-- TODO: Author list -->},
  title   = {Alumet Insight: A Python Tool for Interactive Exploration of
             Alumet-Agent Energy and Resource Measurements},
  journal = {Journal of Open Research Software},
  year    = {<!-- TODO -->},
  volume  = {<!-- TODO -->},
  doi     = {<!-- TODO: JORS DOI -->}
}
```

## Contributing

Contributions are welcome. The most useful ways to contribute are:


Reporting bugs — open an Issue
with a minimal reproducible example including your Alumet-agent version and
a sample (or mock) of the CSV that triggers the problem.
Requesting features — open an Issue describing your use case and which
metric or workflow you would like to see supported.
Submitting code — fork the repository, create a feature branch, and open
a Pull Request. Please include a brief description of the change and, where
applicable, add or update tests.
Improving documentation — edits to README.md or docs/ are very
welcome, especially worked examples using different Alumet-agent metric types.

<!-- TODO: Add a CONTRIBUTING.md with coding standards, branch naming conventions, and review process. -->

## Contact

For questions, bug reports, or general feedback, please open an
Issue on GitHub.
For private enquiries, contact the corresponding author at

<!-- TODO: Add email address -->

## License

Alumet Insight is released under the MIT License. 
<!-- TODO: Add LICENSE -->

## Acknowledgments

The authors thank the Alumet-agent development team
for the underlying measurement framework and for prompt answering to our created issues. 
We would also like to acknowledge RWTH MBD for the use of GAIA GPU cluster.
These materials were initially prepared following version 1 of the [One Good Tutorial software documentation checklist](10.5281/zenodo.19338407).

## Competing Interests

The authors have no competing interests to declare.

## References

<!-- TODO: Alumet-agent primary reference -->


<!-- TODO: Plotly / Dash reference -->

```bibtex
@misc{onegoodtutorial1.0.1,
  author       = {Williams, Peter K. G.},
  title        = {One Good Tutorial (version 1.0.1)},
  year         = 2026,
  publisher    = {Zenodo},
  version      = {1.0.1},
  doi          = {10.5281/zenodo.19338407},
  url          = {https://doi.org/10.5281/zenodo.19338407}
}
```
