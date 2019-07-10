#!/bin/bash
cd "$(dirname "$0")"

(echo cmdline "$0" "$@"; PYTHONPATH=.. python3 train.py "$@" 2>&1 ) | tee "runs/$(date -Is).log"
