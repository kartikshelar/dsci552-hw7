# Homework 7 — Multi-Label SVMs, K-Means and Hierarchical Clustering

DSCI-552 (Machine Learning for Data Science) — Homework 7.

## Structure

```
.
├── README.md
├── requirements.txt
├── Homework7.pdf
├── Homework 7 Data/
│   └── Anuran Calls (MFCCs)/
│       ├── Frogs_MFCCs.csv
│       └── Readme.txt
├── Shelar_Kartik_HW7.ipynb
└── Shelar_Kartik_HW7.pdf
```

## Contents

- `Homework 7 Data/` — extracted contents of the course's `Homework 7 Data.zip`
  (the zip itself is not in the repository). The notebook reads the files from
  this folder.
  - Anuran Calls (MFCCs) (UCI): 7,195 syllables described by 22 MFCC
    attributes, each with three labels: Family (4 classes), Genus (8 classes)
    and Species (10 classes).
- `Shelar_Kartik_HW7.ipynb` — full analysis:
  - **1. Multi-class and multi-label classification with SVMs** (70% random
    training split): exact match ratio, Hamming loss and Hamming score; one
    Gaussian-kernel SVM per label with the penalty and kernel width chosen by
    10-fold cross-validation, with standardized and with raw attributes;
    L1-penalized linear SVMs; and the same with SMOTE to handle class
    imbalance, with a written comparison of the classifiers.
  - **2. K-means on a multi-class, multi-label data set**: 50 Monte Carlo
    repetitions, the number of clusters chosen with the Calinski–Harabasz
    index, majority labels per cluster, and the average Hamming distance,
    Hamming score and Hamming loss with their standard deviations.
  - **3. ISLR exercise 12.6.2**: complete and single linkage dendrograms for
    the given dissimilarity matrix, the two-cluster cuts, and an equivalent
    dendrogram with repositioned leaves.
- `Shelar_Kartik_HW7.pdf` — the notebook printed to PDF (15 pages).
- `Homework7.pdf` — assignment.

## Setup

```bash
pip install -r requirements.txt
```

Then open `Shelar_Kartik_HW7.ipynb` in Jupyter, keeping `Homework 7 Data/` in
the same folder. A full run takes about 30 minutes, mostly the grid searches
for the Gaussian-kernel SVMs, the SMOTE grid search, and the 50 k-means
repetitions in problem 2.
