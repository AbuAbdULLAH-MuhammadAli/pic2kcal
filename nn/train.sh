#!/bin/bash
cd "$(dirname "$0")"
PYTHONPATH=.. python3 train.py
