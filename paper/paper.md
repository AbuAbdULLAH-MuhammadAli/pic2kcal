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
  
citation-style: template/ieee-with-url.csl
link-citations: true
---

# Motivation and Related Work

There's some other papers like [@chokr; @takumi; @salvador].

# Results

|                       | relative error | L1 of g / 100g    |         |             |
| --------------------- | -------------- | ----------------- | ------- | ----------- |
| **kcal**              |                | **carbohydrates** | **fat** | **protein** |
| baseline              | 0.464          | 10.5g             | 4.5g    | 3.1g        |
| ours (kcal only)      | 0.361          | ---               | ---     | ---         |
| ours (w/ macros)      | 0.352          | 7.9g              | 4.1g    | 2.7g        |
| ours (w/ macros+ings) | 0.328          | 7.1g              | 3.9g    | 2.5g        |

: Results per 100g. Note multitask learning improves performance.

# References

