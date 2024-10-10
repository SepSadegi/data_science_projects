'''We learn how to interpret a DNA structure and how machine learning
algorithms can be used to build a prediction model on DNA sequence data.
DNA data handling using Biopython

Key Features of Biopython:
 - Sequence Handling: Reading and writing sequence file formats (e.g., FASTA, GenBank).
 - Sequence Analysis: Functions for calculating sequence statistics, performing alignments, and analyzing motifs.
 - 3D Structure Handling: Tools for working with 3D molecular structures, including parsing PDB files.
 - Phylogenetics: Tools for creating and manipulating phylogenetic trees.
 - Bioinformatics Databases: Access to online databases like NCBI, and tools for fetching and parsing data from these databases.
 - Machine Learning:  Interfaces to apply machine learning techniques to biological data.
'''

import numpy as np
import pandas as pd
import seaborn as sns
from Bio import SeqIO
import matplotlib.pyplot as plt

# for sequence in SeqIO.parse('data/example_dna.fa', "fasta"):
#     print(sequence.id)
#     print(sequence.seq)
#     print(len(sequence))

# Read the text files into a DataFrame
human_df = pd.read_table('../data/human.txt')
print(human_df.head())
print(human_df.shape)
print(human_df.columns)

chimpanzee_df = pd.read_table('../data/chimpanzee.txt')
print(chimpanzee_df.head())
print(chimpanzee_df.shape)
print(chimpanzee_df.columns)

dog_df = pd.read_table('../data/dog.txt')
print(dog_df.head())
print(dog_df.shape)
print(dog_df.columns)

# Determine the percentage of each class in the Human DataFrame
# Get the unique classes
human_classes = human_df['class'].unique()
print(human_classes)

# Count number of sequences in each class
human_class_counts = human_df['class'].value_counts()
print(human_class_counts)

# Calculate the percentage of the presence of each class
human_class_percentages =  human_class_counts / human_class_counts.sum() * 100
print(human_class_percentages)

labels = [
    "6: Transcription factor",
    "4: Synthase",
    "3: Synthetase",
    "1: Tyrosine kinases",
    "0: G protein-coupled receptors",
    "2: Tyrosine phosphatase",
    "5: Ion channels"]

# Define colors for the pie chart
cmap = plt.get_cmap("Dark2") # Choose a colormap (e.g., 'tab10', 'Accent', 'paired', 'Dark2')

# Create the pie chart
plt.figure(figsize=(10, 6))
plt.pie(human_class_percentages, labels=labels, autopct="%1.1f%%", startangle=90, colors=cmap(np.arange(len(labels))))

# Customize the pie chart design
plt.title("Distribution of Protein Class Types in Human Genome Data", fontweight="bold")
plt.xlabel("Class")
plt.ylabel("Percentage")
plt.axis("equal") # Equal aspect ratio ensures that the pie chart is circular
# plt.legend(title="Classes", loc="upper right")
plt.tight_layout()
plt.savefig('human_protein_class_analysis_chart.jpg', format='jpg', dpi=300)
plt.show()
