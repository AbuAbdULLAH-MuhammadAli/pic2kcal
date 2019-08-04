---
title: "pic2kcal: End-to-End Calorie Estimation From Pictures of Food"
author: |
    Robin Ruede, Lukas Frank, Verena Heußer \
    Karlsruhe Institute of Technology
date: 2019-08-02
abstract: |
    We estimate kcal directly from a picture.
    It good.

citekeys:
    chokr: https://dl.acm.org/citation.cfm?id=3297871
    takumi: https://dl.acm.org/citation.cfm?doid=3126686.3126742 # http://img.cs.uec.ac.jp/pub/conf17/171024ege_0.pdf
    salvador: https://arxiv.org/abs/1812.06164

citation-style: template/ieee.csl
link-citations: true
---

# Introduction

# Related Work

There's some other papers like [@chokr; @takumi; @salvador]. Ours is more end to end and also BETTER

# Dataset Extraction and Preprocessing

## Matching / Preprocessing

-   preprocessing / fddb matching

## Dataset Statistics

-   statistics about dataset
-   number of pictures per recipe
-   number of recipes
-   most common ingredients

# Models

- (SmoothL1Loss + weight * BCE)
- What we predict (kcal, fat, etc, ings)

# Experiments

- todo: actually test 

\clearpage

# Results

Our results can be seen in [@tbl:res]. Example outputs can be seen in [@fig:results].

\begin{table}
\begin{center}
\begin{tabular}{|l|c|}
\hline
Method & kcal relative error \\
\hline\hline
baseline & 0.464 \\
ours (kcal only) & 0.361 \\
ours (w/ macros) & 0.352 \\
ours (w/ macros+ings) & 0.328 \\
\hline
\end{tabular}
\end{center}
\caption{Results per 100g. Note that multitask learning improves performance.\label{tbl:res}}
\end{table}

![Some example results, showing predicted calories, fat, protein, carbohydrates and ingredients.](img/results.png){#fig:results}

# References
