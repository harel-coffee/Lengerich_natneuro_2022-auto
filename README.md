# Lengerich Natneuro 2022

This repository hosts the code used to analyze both transcriptomic and morphology imaging data
in the paper "" by Lengerich et al, 2022. 

## Transcriptomics
R code for transcriptomics can be found in the "transcriptomics" folder as a series of R Markdown files. The output of each R Markdown is used as input to the next R Markdown file. If you wish to regenerate the single-cell analyses from scratch, please download the raw data files from GEO and process them using Cell Ranger as described in the paper. The Cell Ranger outputs are then used in the first R Markdown file in each Study subfolder. If you wnat to skip these early processing steps and regerate the final anlyses directly, we have also uploaded to GEO a counts matrix and cell-level metadata table for each single-cell study. Together, these can be used to build the exact final Seurat objects used to make figures.

R Packages and the versions used are provided in the `renv.lock` file. This can be used with the `renv` package to reproduce the environment used in this analysis.

## Morphology

Python code for morphology can be found in the "morphology" folder.
