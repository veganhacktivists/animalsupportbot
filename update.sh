#!/bin/bash

echo "Getting knowledge csv files from gdrive..."
python update_knowledge.py || exit

echo "Computing template dictionaries..."
python argmatcher.py || exit

