#!/bin/bash
#
# This script converts all Jupyter notebooks in the ../Notebooks directory
# into Python (.py) scripts and moves them into the current directory (src/).

# Find all .ipynb files in the ../Notebooks directory (without descending into subdirectories)
# and convert them to Python scripts in parallel.
# -maxdepth 1: Prevents find from descending into subdirectories.
# -name '*ipynb': Matches all files ending with .ipynb.
# xargs -t -P2: Executes the command for each input, printing the command (-t)
#               and using up to 2 parallel processes (-P2).
# jupyter nbconvert --to python: The command to perform the conversion.
find ../Notebooks -maxdepth 1 -name '*ipynb' | xargs -t -P2 jupyter nbconvert --to python

# Find all newly created .py files in the ../Notebooks directory and move them
# to the current directory (src/).
# -print0 / -0: Handles filenames with spaces or special characters correctly.
# -I%: Replaces '%' with the input filename.
find ../Notebooks -maxdepth 1 -name '*py' -print0 | xargs -0 -t -I% mv % .
