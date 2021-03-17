#!/usr/bin/env bash

# Look for installation instructions in README.md

. venv/bin/activate
pip install wheel
pip install -r requirements.txt
# link ops package instead of copying
# jupyter and snakemake will import code from .py files in the ops/ directory
pip install -e .

