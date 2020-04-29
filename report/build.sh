#!/bin/bash

pandoc paper.md -o paper.pdf --filter=pandoc-crossref --filter=./getcite/index.js --filter=pandoc-citeproc --template=template/egpaper_final.tex
