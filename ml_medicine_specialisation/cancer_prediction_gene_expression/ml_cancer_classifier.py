'''
Machine Learning for Cancer Prediction

The data used for this tutorial is an RNA-seq gene expression data for different cancer types.
The rows represent cancer samples and the columns represent gene count values. The last column
contains the cancer categories.

The original data can found here: https://archive.ics.uci.edu/dataset/401/gene+expression+cancer+rna+seq

In the small dataset 8000 genes are presented with the following cancer types:

BRCA    BRCA1 and BRCA2 Genes: breast cancer, ovarian cancer
KIRC    Kidney renal clear cell carcinoma
LUAD    Lung adenocarcinoma
PRAD    Prostate adenocarcinoma
COAD    Colon adenocarcinoma

In the large dataset 20530 genes are presented, labels for cancer types are in the labels file

This is an example of supervised learning using RandomForestClassifier.
This algorithm is used for classification tasks where the goal is to predict discrete labels.

'''

import os
# data handling
import pandas as pd
import numpy as np

# data visualization
import matplotlib.pyplot as plt
import matplotlib as mpl


# Configure matplotlib
mpl.rc('xtick', labelsize=16)
mpl.rc('ytick', labelsize=16)
mpl.rcParams.update({'font.size': 15})
plt.ioff()

# Define the path to the dataset
data_directory = '../cancer_gene_expression'

# Flags for plotting sections
plot_image = False

# Load the small dataset
df = pd.read_csv(os.path.join(data_directory, 'cancer_gene_expression.csv'))
# Load the large dataset
# original_df = pd.read_csv(os.path.join(data_directory, 'data.csv'))
# labels_df = pd.read_csv(os.path.join(data_directory, 'labels.csv'))


# Data Exploration & Cleaning
# print(df.shape) # (801, 8001)
# print(original_df.shape) # (801, 20532)
# print(labels_df.shape) # (801, 2)

# print(df.columns)
# print(original_df.columns)
# print(labels_df.columns)
# print(df.info)
# print(labels_df.info)
# print(df['Cancer_Type'].head())
# print(labels_df['Class'].head())
# print(original_df['Unnamed: 0'].head())

# Check how many different cancer types are there in the data
counts = df['Cancer_Type'].value_counts()
print(counts)
# print(labels_df['Class'].value_counts())
# print(original_df['Unnamed: 0'].value_counts())

# print(df.head(1))
# print(original_df.iloc[:,2])
# print(labels_df.head(1))
# print(df['gene_2'])

# Check for missing values
datanul = df.isnull().sum()
missing_cols = datanul[datanul > 0]
print(f"Columns with missing values: {len(missing_cols)}")

if plot_image:
    plt.figure(figsize=(10, 8))
    plt.bar(counts.index, counts.values)
    plt.title("Distribution of Cancer Types")
    plt.xlabel('Cancer Type')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

# Data preprocessing