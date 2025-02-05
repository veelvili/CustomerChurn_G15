# -*- coding: utf-8 -*-
"""EDA (1).ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1ahykUloJsGTrkoGE-vcDkeOPNtB0oJAv

## Data Cleaning, Preprocessing and EDA for Bank Customer Churn Prediction Dataset
"""

import numpy as np
import pandas as pd

"""### Load the Dataset"""

# Load the dataset
df = pd.read_csv('Customer-Churn-Records.csv')
# Check the number of rows and columns
print("Shape of the dataset:", df.shape)
print(df.columns)

"""### Check for Null Values & Data Types"""

# Display the counts in the desired format
for dtype, count in  df.dtypes.value_counts().items():
    print(f"Columns of datatype {dtype}: {count}")

# check number of columns with null values
null_counts = df.isnull().sum()
print(f"Total number of columns with null values: {len((null_counts[null_counts > 0]).tolist())}/{len(df.columns.tolist())}")

# Create DataFrame with column names, data types, and null counts
column_info = pd.DataFrame({
    "Column Name": df.columns,
    "Data Type": [df[col].dtype for col in df.columns],
    "Null Counts":  df.isnull().sum().values
})
column_info.index = column_info.index + 1
column_info

"""### Convert all data to numerical format for data modeling"""

# get columns of object data type only
object_columns = df.select_dtypes(include='object').columns

# check the unique values and their total number for each object column
for col in object_columns:
    unique_values = df[col].unique()
    print(f"Unique values in '{col}': {unique_values}")
    total_num_of_unique_values =  df[col].nunique()
    print(f"Total number of Unique values in '{col}': {total_num_of_unique_values}\n")

# we first drop the Surname and Row Number columns from the DataFrame
df = df.drop(columns=['Surname', "RowNumber"])

# verify Surname column is removed
df.columns

from sklearn.preprocessing import LabelEncoder

# use label encoding for Gender column as it contains categorical values that is not
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])

# use One-Hot encoding Geography column since there is no particular orders in the value
df = pd.get_dummies(df, columns=['Geography'])

card_type_mapping = {
    'SILVER': 1, # lowest in rank
    'GOLD': 2,
    'PLATINUM': 3,
    'DIAMOND': 4  # highest in rank
}
df['Card Type'] = df['Card Type'].map(card_type_mapping)

# to verify the data transformation
column_data_types = pd.DataFrame(df.dtypes, columns=['Data Type']).reset_index()
column_data_types.columns = ['Column Name', 'Data Type']  # Rename columns

column_data_types.index = column_data_types.index + 1
column_data_types

# Convert Boolean values to 1 and 0 in Geography columns (uncomment the below if your need to use this line of code)
# df[['Geography_France', 'Geography_Germany', 'Geography_Spain']] = df[['Geography_France', 'Geography_Germany', 'Geography_Spain']].astype(int)

"""### Scaling & Normalization"""

# get columns of int and float data type only
int_and_float_columns_df = df[df.select_dtypes(include=['int64', 'float64']).columns]

# check their value range
int_and_float_columns_df.head()

# Select the columns of interest
columns_to_scale = ['CreditScore', 'Age', 'Tenure', 'Balance', 'EstimatedSalary', 'NumOfProducts', 'Point Earned']

# Get summary statistics for these columns
summary_stats = df[columns_to_scale].describe().loc[['min', 'max', 'mean', 'std']]
summary_stats

# Define a dictionary to store outlier counts for each column
outlier_counts = {}

for column in columns_to_scale:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Identify outliers
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    outlier_counts[column] = len(outliers)

# Display the count of outliers in each column
print(outlier_counts)

# apply winsorizing method for Number of product column
df['NumOfProducts'] = np.clip(df['NumOfProducts'], df['NumOfProducts'].quantile(0.05), df['NumOfProducts'].quantile(0.95))

# Calculate Z-scores for CreditScore & filter out the rows where Z-score is greater than the threshold
df['CreditScore_Z'] = (df['CreditScore'] - df['CreditScore'].mean()) / df['CreditScore'].std()
threshold = 3
df_filtered = df[np.abs(df['CreditScore_Z']) <= threshold]

# Drop the temporary Z-score column
df_filtered = df_filtered.drop(columns=['CreditScore_Z'])

df_filtered.shape

# Calculate the correlation between 'Age' and 'Exited'
correlation_age_exited = df_filtered['Age'].corr(df_filtered['Exited'])

# Print the result
print(f"Correlation between Age and Exited: {correlation_age_exited}")

# Define the IQR bounds for the Age column to identify outliers
Q1 = df_filtered['Age'].quantile(0.25)
Q3 = df_filtered['Age'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filter rows where Age is considered an outlier
age_outliers = df_filtered[(df_filtered['Age'] < lower_bound) | (df_filtered['Age'] > upper_bound)]

# Calculate the proportion of Exited vs. Non-Exited among Age outliers
age_outliers_exited_counts = age_outliers['Exited'].value_counts()
total_outliers = len(age_outliers)
exited_proportion_in_outliers = age_outliers_exited_counts / total_outliers

print("\nProportion of 'Exited' within Age outliers:")
print(exited_proportion_in_outliers)

# Get the count of each class in the Exited column for comparison
class_counts = df_filtered['Exited'].value_counts()
total_number_of_samples = len(df_filtered)
exited_proportion_overall = class_counts / total_number_of_samples

print("\nOverall Proportion of 'Exited':")
print(exited_proportion_overall)

# Define the lower and upper percentiles for capping
lower_percentile = 1
upper_percentile = 99

# Calculate the lower and upper bounds using the percentiles
lower_bound = np.percentile(df_filtered['Age'], lower_percentile)
upper_bound = np.percentile(df_filtered['Age'], upper_percentile)

# Apply capping by replacing values below lower bound with the lower bound & values aboveupper bound with the upper bound
df_filtered['Age'] = np.where(df_filtered['Age'] < lower_bound, lower_bound, df_filtered['Age'])
df_filtered['Age'] = np.where(df_filtered['Age'] > upper_bound, upper_bound, df_filtered['Age'])

print(df_filtered['Age'].describe())

# verify that the proportion of Exited vs. Non-Exited remains at ~ 80/20 split after the scaling
class_counts = df_filtered['Exited'].value_counts()
total_number_of_samples = len(df_filtered)
exited_proportion_overall = class_counts / total_number_of_samples

print("\nOverall Proportion of 'Exited':")
print(exited_proportion_overall)

"""### Feature Selection"""

import seaborn as sns
import matplotlib.pyplot as plt

# Calculate the correlation matrix for the numerical features
correlation_matrix = df_filtered.corr()

# Visualize the correlation matrix using a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0, cbar=True)
plt.title("Correlation Matrix")
plt.show()

# Identify highly correlated features (e.g., with a correlation coefficient > 0.8 or < -0.8)
threshold = 0.8
highly_correlated = np.where((correlation_matrix > threshold) | (correlation_matrix < -threshold))

# Extract the indices of highly correlated features (excluding the diagonal)
highly_correlated_pairs = [(correlation_matrix.columns[x], correlation_matrix.columns[y])
                           for x, y in zip(*highly_correlated)
                           if x != y and x < y]  # to avoid duplicate pairs

# Display highly correlated feature pairs
print("\nHighly Correlated Feature Pairs (correlation > 0.8 or < -0.8):")
for pair in highly_correlated_pairs:
    print(pair)

"""### Analyze Class Imbalance"""

# calculate the proportions of each class for the target variables
class_proportions = df_filtered['Exited'].value_counts(normalize=True) * 100
print(class_proportions)

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from collections import Counter

# Define the predicting variable & the target variable
X = df_filtered.drop(columns=['Exited'])
y = df_filtered['Exited']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply SMOTE only to the training data
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Verify the class distribution
print("Class distribution after SMOTE:", Counter(y_train_smote))

"""### Output Processed Data as CSV"""

# Convert to DataFrames and save as CSVs
train_df = pd.concat([X_train_smote, y_train_smote], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)

train_df.to_csv("bank_customer_churn_train_data_processed.csv", index=False)
test_df.to_csv("bank_customer_churn_test_data_processed.csv", index=False)