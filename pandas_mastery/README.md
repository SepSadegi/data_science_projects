# Pandas Mastery

This repository is dedicated to mastering the core structures of pandas—an essential Python library for data 
manipulation and analysis. The goal is to cover key aspects of **DataFrames** and **Series** with clear explanations, 
practical examples, and engaging challenges.

The content is inspired by resources such as the [Kaggle Pandas Course](https://www.kaggle.com/learn/pandas) and 
the [Pandas Series by Matin Mahmoudi](https://github.com/matinmahmoudi/Kaggle/tree/5ffe3bd6dcad7eed5031f7598ef4f837449d41e0).

---

## Table of Contents
1. [Introduction to DataFrames](#1-introduction-to-dataframes)
2. [DataFrame Creation](#2-dataframe-creation)
3. [Working with CSV Files](#3-working-with-csv-files)
4. [DataFrame Inspection](#4-dataframe-inspection)
5. [DataFrame Indexing](#5-dataframe-indexing)
6. [DataFrame Operations and Methods](#6-dataframe-operations-and-methods)
7. [DataFrame Aggregation](#7-dataframe-aggregation)
8. [Merging DataFrames](#8-merging-dataframes)
9. [GroupBy Operations](#9-groupby-operations)
10. [Pivoting, Reshaping, and Other Operations](#10-pivoting-reshaping-and-other-operations)

---

## 1. Introduction to DataFrames

A **DataFrame** is a two-dimensional, mutable data structure in pandas that holds rows and columns, similar to a 
table in a database or a spreadsheet. It is one of the most powerful tools for working with data in pandas.

### Key Concepts:
- **Rows**: Represent individual records or observations.
- **Columns**: Represent different variables or features of the data.

### Features of DataFrames:
- **Heterogeneous Data**: Columns can contain different types of data (integers, floats, strings, etc.).
- **Labeled Axes**: Both rows and columns have labels (i.e., indices).
- **Size-Mutable**: The shape (rows and columns) of a DataFrame can be modified dynamically (i.e., expanded or contracted).
- **Indexing**: Data can be accessed using both label-based and position-based indexing.
- **Flexible Input/Output**: DataFrames can be easily created from or written to formats like CSV, Excel, SQL databases, 
and more.

---

## 2. DataFrame Creation

There are several common ways to create DataFrames in pandas:

- **From Dictionary**: Keys represent column names, and values are lists or arrays of column data.
- **From List of Lists**: Each list represents a row, and column names are specified.
- **From List of Dictionaries**: Each dictionary in a list represents a row of data.
- **From Another DataFrame**: DataFrames can be copied and modified independently. This is useful when we want to manipulate 
data without modifying the original DataFrame.
- **Using `from_records()`**: Converts structured or record arrays into DataFrames.
- **Reading from CSV or Other Files**: Data can be loaded from structured files like CSV, Excel, or SQL.

The code for creating and saving DataFrames is located in `dataframe_creation_saving.py`.

---

## 3. Working with CSV Files

CSV (Comma-Separated Values) files are a common format for storing tabular data. Pandas makes it easy to read from 
and write to CSV files.

- **Exporting to CSV**: Use the `DataFrame.to_csv()` method to save a DataFrame as a CSV file. Options include the 
file path, delimiter, and whether to include the index.
- **Reading from CSV**: Use the `pandas.read_csv()` function to load data from a CSV file into a DataFrame.

The code for exporting DataFrames is located in `dataframe_creation_saving.py`, and the code for reading CSVs can be found in `dataframe_inspection_indexing.py`.

---

## 4. DataFrame Inspection

Inspecting a DataFrame is crucial for understanding its structure and content.

### Common Inspection Methods:
- `df.info()` – Provides a summary of the DataFrame, including index dtype, column dtypes, non-null values, and memory usage.
- `df.shape` – Returns the dimensions of the DataFrame (number of rows, number of columns).
- `df.head()` – Displays the first few rows of the DataFrame.
- `df.tail()` – Displays the last few rows of the DataFrame.
- `df.size` – Returns the total number of elements in the DataFrame.
- `df.dtypes` – Lists the data types of each column.
- `df.columns` – Returns the column labels.
- `df.axes` – Returns the row and column axis labels.
- `df.describe()` – Provides summary statistics (mean, standard deviation, etc.) for numeric columns.

The code for DataFrame inspection is provided in the `dataframe_inspection_indexing.py` script.

---

## 5. DataFrame Indexing

Indexing in pandas refers to selecting and accessing data from a DataFrame using labels, integer locations, or 
conditions. It allows efficient data selection based on rows, columns, or a combination of both.

### Key Indexing Methods:
- **Selecting Columns**: 
  Example: `df['column_name']` or `df[['col1', 'col2']]`
 
- **Selecting Rows**: 
  - `.loc[]` – Label-based indexing using row/column names. 
    Example: `df.loc[0, 'Name']`
  - `.iloc[]` – Position-based indexing using integer positions. 
    Example: `df.iloc[0, 1]`
    
- **Boolean Indexing**: Selecting data based on conditions. 
  Example: `df[df['Age'] > 100]`
  
- **Slicing**: Selecting a range of rows or columns. 
  Example: `df.loc[2:4]` or `df.iloc[0:3]`
   
### Difference Between `iloc` and `loc`:

- **`iloc` (Integer Location)**: 
  - Position-based indexing.
  - Uses Python's standard indexing (first element included, last excluded). So `0:10` selects entries `0,...,9`.
  - Example: `df.iloc[0, 1]` selects the element in the first row and second column.

- **`loc` (Label Location)**: 
  - Label-based indexing.
  - Indexes inclusively, meaning both the start and end of a range are included. So `0:10` selects entries `0,...,10`.
  - Example: `df.loc[0, 'Name']` selects the element in the row with index label `0` and column `'Name'`.
  
  **Note**: A key difference between `iloc` and `loc` is the way ranges are handled:
  - `iloc[0:10]` selects 10 entries (0 to 9).
  - `loc[0:10]` selects 11 entries (0 to 10).
- This can be confusing when working with numerical indices. For example, `df.iloc[0:1000]` returns 1000 entries, 
but `df.loc[0:1000]` returns 1001 entries.

The code for indexing is located in `dataframe_inspection_indexing.py`.

---

## 6. DataFrame Operations and Methods

The following operations on DataFrames are handy for data manipulation:

- **Inserting Columns**: 
  Add new columns by assigning values. 
  Example: `df['NewColumn'] = value_list`
  
- **Removing Columns**: 
  Use the `drop()` method to remove columns. 
  Example: `df.drop('ColumnName', axis=1, inplace=True)`

- **Updating Values**: 
  Modify values using indexing or conditions.  
  Example: `df.loc[df['Column'] > 50, 'Column'] = 100`
  
- **Renaming Columns**:  
  Rename columns using the `rename()` method or modify the `columns` attribute.  
  Example: `df.rename(columns={'OldName': 'NewName'}, inplace=True)`

The code for performing these operations is located in `dataframe_basic_operations.py`.

---

## 7. DataFrame Aggregation

Aggregation operations are used to summarize data in a DataFrame. Common methods include:
- `df.sum()`, `df.mean()`, `df.median()`, `df.min()`, `df.max()`, `df.count()`, and `df.agg()` for custom aggregations.

---

## 8. Merging DataFrames

There are multiple ways to merge or combine DataFrames in pandas:
- **`pd.concat()`**: Concatenate DataFrames along rows or columns.
- **`pd.merge()`**: Perform SQL-like merges (inner join, outer join, etc.).

### Merge Types:
- **Inner Join**: Retains only the common rows.
- **Left Join**: Retains all rows from the left DataFrame.
- **Right Join**: Retains all rows from the right DataFrame.
- **Outer Join**: Retains all rows from both DataFrames.

---

## 9. GroupBy Operations

The `groupby()` function is used to group data based on column values and perform aggregate operations on the grouped data. 
Example: `df.groupby('column_name').sum()`

---

## 10. Pivoting, Reshaping, and Other Operations

- **Pivoting**: Use `df.pivot_table()` to reorganize data.
- **Reshaping**: Use `df.melt()`, `df.stack()`, and `df.unstack()` to reshape DataFrames.
- **Filtering Data**: Apply conditions to filter rows.
- **Handling Missing Data**: Use `df.fillna()` or `df.dropna()` to manage missing data.













    
    
    
    

