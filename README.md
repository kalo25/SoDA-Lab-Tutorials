# Voter Turnout Analysis (Reproducible Template — Python)

## Overview
This project analyzes voter turnout patterns using a fully reproducible pipeline.

## Reproducibility
- Outputs are saved in `outputs/` (figures, tables, environment info).
- Package snapshot is saved via `pip freeze` to `outputs/requirements_frozen.txt`.

## Installation
1. Create and activate a virtual environment (recommended).
2. Install dependencies:
```bash
pip install numpy pandas matplotlib statsmodels
```

## Usage
Run the analysis script:
```bash
python reproducibility.py
```

## Docker Usage
```bash
docker build -t voter-analysis-py .
docker run -v $(pwd)/outputs:/project/outputs voter-analysis-py
```

## Output
- Figures: `outputs/figures/`
- Tables: `outputs/tables/`
- Environment: `outputs/python_environment.txt`, `outputs/requirements_frozen.txt`
