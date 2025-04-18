

# RNCQR

[![Python Version](https://img.shields.io/badge/python-3.7%2B-blue)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()

## Overview
This repository implements an RNCQR model-based simulation system containing data generation, model training, and hyperparameter configuration modules.

## Table of Contents
• [Installation](#installation)
• [Quick Start](#quick-start)
• [Project Structure](#project-structure) 
• [Configuration](#configuration)
• [License](#license)

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

Recommended to use Python 3.7+ with virtualenv for environment isolation.

## Quick Start

Execute the main simulation script:
```bash
python Run_simulation.py
```

## Project Structure

```
├── Data.py               # Synthetic data generation
├── Hyperparameters.py    # Hyperparameter loader (provides get_hyperparameters())
├── loss_function.py      # Custom loss function implementation
├── Model.py              # RNCQR model architecture & training algorithm
├── Utils.py              # Utility functions for common operations
├── parameters.json       # JSON configuration for all hyperparameters
├── Run_simulation.py     # Main entry point for simulation experiments
└── requirements.txt      # Python dependencies specification
```

## Configuration

Modify `parameters.json` to adjust experiment settings
## License
This project is licensed under the MIT License - see the LICENSE file for details



