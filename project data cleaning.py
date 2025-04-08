import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Load the CSV file
df = pd.read_csv("C:/Users/parth/Downloads/INC 5000 Companies 2019.csv")

df.info(), df.head()

# Starter script for Data Cleaning & Preparation

#  Make a copy of the original data
df_clean = df.copy()

# Clean revenue column - remove "Million", "$", and convert to float (in millions)
df_clean['revenue'] = df_clean['revenue'].str.replace(' Million', '', regex=False)
df_clean['revenue'] = df_clean['revenue'].str.replace('$', '', regex=False)
df_clean['revenue'] = pd.to_numeric(df_clean['revenue'], errors='coerce')

# For now, we fill missing 'workers' with median value (more robust than mean)
# Fix for workers
df_clean['workers'] = df_clean['workers'].fillna(df_clean['workers'].median())

# Fix for metro
df_clean['metro'] = df_clean['metro'].fillna('Unknown')

#Create new derived columns
df_clean['company_age_2019'] = 2019 - df_clean['founded']

# Check cleaned data
df_clean.info() 
df_clean.head()