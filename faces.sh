#!/bin/bash

# Script for executing the Python version of Faces
# Usage:
# bash faces.sh <training_file> <facit_file> <test_file>

# Author: Ola Ringdahl

# the location of this script:
base_dir="$(dirname "$0")"

# if you are using Python2, just change to python2 below
python3 $base_dir/faces.py $1 $2 $3

