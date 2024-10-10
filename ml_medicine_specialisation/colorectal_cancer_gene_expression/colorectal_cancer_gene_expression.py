import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rc('xtick', labelsize=16)  # 16
mpl.rc('ytick', labelsize=16)
mpl.rcParams.update({'font.size': 15}) # 25 for 20:10 18
plt.ioff()

datapath = '../Colorectal_Cancer_Dataset'

# Flags to control individual plotting sections
plot_patient_info = False
save_to_file = False
plot_basic_statistics = False

patient_df = pd.read_csv(os.path.join(datapath, 'colorectal_cancer_patient_data.csv'))
# print(patient_df.head())
# print(patient_df.shape)
# print(patient_df.columns)

if plot_patient_info:
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle('Histogram of Patient Data')

    #List of columns to plot
    columns = ['Age (in years)', 'Dukes Stage', 'Gender', 'Location', 'DFS (in months)', 'DFS event', 'Adj_Radio', 'Adj_Chem']
    colors = ["Red", "Orange", "Gold", "Limegreen", "Mediumseagreen", "Darkturquoise", "Steelblue", "Purple"]

    # Iterate through columns and create histograms
    for i, col in enumerate(columns):
        row = i // 4
        col_num = i % 4
        ax = axes[row, col_num]

        if patient_df[col].dtype == 'object': # Categorical data
            sns.countplot(x=col, data=patient_df, ax=ax, color=colors[i])
        elif patient_df[col].nunique() == 2: # Binary data (0 and 1)
            sns.countplot(x=col, data=patient_df, ax=ax, color=colors[i])
        else: # Numerical data
            hist = sns.histplot(patient_df[col], bins=5, kde=True, ax=ax, color=colors[i])

            # Calculate bin centers
            bin_edges = hist.patches[0].get_x() + hist.patches[0].get_width() * np.arange(len(hist.patches) + 1)
            bin_centers = bin_edges + hist.patches[0].get_width() / 2

            # ax.set_xticks(bin_edges)
            # ax.set_xticklabels([f'{center:.1f}' for center in bin_centers])

        ax.set_title(col)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    # plt.savefig('colorectal_cancer_patient_data_distribution.jpg', format='jpg', dpi=300)
    plt.show()

gene_df = pd.read_csv(os.path.join(datapath, 'colorectal_cancer_gene_expression_data.csv'))
# print(gene_df.head())
# print(gene_df.shape)
# print(gene_df.columns)
# print(gene_df['ID_REF'])

# Drop the '#' column
gene_df = gene_df.drop("#", axis=1)

# Transpose to align with the patients' data
gene_df = gene_df.transpose()
# print(gene_df.head())
# print(gene_df.shape)

# Use the first row as column header
column_name = gene_df.iloc[0].tolist()
gene_df.columns = column_name
# Remove the redundant first row
gene_df = gene_df.drop("ID_REF", axis=0)
# print(len(column_name))
# print(column_name)
# print(gene_df.columns)
# print(gene_df.head())
# print(gene_df.shape)

numeric_gene_features = gene_df.columns.tolist()[1:]
gene_df[numeric_gene_features] = gene_df[numeric_gene_features].astype(float)
# print(numeric_gene_features)
# Basic statistic
gene_summary_stats = gene_df[numeric_gene_features].describe()
print(gene_summary_stats)
if save_to_file:
    gene_summary_stats.to_csv('gene_summary_statistics.csv')
