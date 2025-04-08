import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline 

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


# Set Seaborn style
sns.set(style="whitegrid")

# -------------------------------
# 1. Exploratory Data Analysis (EDA)
# -------------------------------

# Top 10 industries by number of companies
top_industries = df_clean['industry'].value_counts().head(10)
print("Top 10 Industries by Number of Companies: ",top_industries)

# Revenue distribution
revenue_distribution = df_clean['revenue']
print("Revenue Distribution: ", revenue_distribution.describe())

# Company age distribution
company_age_distribution = df_clean['company_age_2019']
print("Company Age Distribution: ", company_age_distribution.describe())

# -------------------------------
# 2. Workforce & Revenue Correlation
# -------------------------------

# Calculate correlation
correlation = df_clean[['workers', 'revenue']].corr()
print("Correlation between Workforce and Revenue: ", correlation)

# -------------------------------
# 3. Geographic Analysis of Growth
# -------------------------------

# Average growth by state
state_growth = df_clean.groupby('state')['growth_%'].mean().sort_values(ascending=False).head(10)
print("Top 10 States by Average Growth: ", state_growth)

# Companies per state
companies_per_state = df_clean['state'].value_counts().head(10)
print("Companies per State: ", companies_per_state)

# -------------------------------
# 4. Revenue vs. Growth Efficiency
# -------------------------------

# Scatter plot data
revenue_growth = df_clean[['revenue', 'growth_%']]

# -------------------------------
# 5. Data Visualization of Key Metrics
# -------------------------------

def plot_all():
    # Top 10 industries
    plt.figure(figsize=(10,5))
    sns.barplot(x=top_industries.values, y=top_industries.index, hue=top_industries.index, palette="viridis", legend=False)
    plt.title("Top 10 Industries by Number of Companies")
    plt.xlabel("Number of Companies")
    plt.tight_layout()
    plt.show()

    # Revenue distribution
    plt.figure(figsize=(8,4))
    sns.histplot(revenue_distribution, bins=50, kde=True, color='blue')
    plt.title("Revenue Distribution")
    plt.xlabel("Revenue (in millions)")
    plt.tight_layout()
    plt.show()

    # Company age distribution
    plt.figure(figsize=(8,4))
    sns.histplot(company_age_distribution, bins=30, kde=True, color='green')
    plt.title("Company Age Distribution (as of 2019)")
    plt.xlabel("Years")
    plt.tight_layout()
    plt.show()

    # Workforce vs Revenue
    plt.figure(figsize=(6,6))
    sns.scatterplot(data=df_clean, x="workers", y="revenue", alpha=0.5)
    plt.title("Workforce vs Revenue")
    plt.xlabel("Number of Workers")
    plt.ylabel("Revenue (in millions)")
    plt.tight_layout()
    plt.show()

    # State vs average growth
    plt.figure(figsize=(10,5))
    sns.barplot(x=state_growth.values, y=state_growth.index, hue=state_growth.index, palette="rocket", legend=False)
    plt.title("Top 10 States by Average Growth")
    plt.xlabel("Average Growth (%)")
    plt.tight_layout()
    plt.show()

    # Revenue vs Growth scatter
    plt.figure(figsize=(6,6))
    sns.scatterplot(data=revenue_growth, x="revenue", y="growth_%", alpha=0.5, color='purple')
    plt.title("Revenue vs Growth Efficiency")
    plt.xlabel("Revenue (in millions)")
    plt.ylabel("Growth (%)")
    plt.tight_layout()
    plt.show()

    # -------------------------------------
# 6. Impact of Founding Year on Growth
# -------------------------------------

# Scatter plot: Founding year vs. Growth
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df_clean, x='founded', y='growth_%', alpha=0.6)
plt.title("Impact of Founding Year on Growth")
plt.xlabel("Founding Year")
plt.ylabel("Growth (%)")
plt.tight_layout()
plt.show()

# Optional: Average growth by founding year
avg_growth_by_year = df_clean.groupby('founded')['growth_%'].mean()
print("Average Growth by Founding Year: ", avg_growth_by_year)

plt.figure(figsize=(10, 5))
sns.lineplot(x=avg_growth_by_year.index, y=avg_growth_by_year.values, marker='o')
plt.title("Average Growth by Founding Year")
plt.xlabel("Founding Year")
plt.ylabel("Average Growth (%)")
plt.tight_layout()
plt.show()

# Founding Year Impact
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df_clean, x='founded', y='growth_%', alpha=0.6)
plt.title("Impact of Founding Year on Growth")
plt.xlabel("Founding Year")
plt.ylabel("Growth (%)")
plt.tight_layout()
plt.show()

avg_growth_by_year = df_clean.groupby('founded')['growth_%'].mean()
print("Average Growth by Founding Year: ", avg_growth_by_year)
plt.figure(figsize=(10, 5))
sns.lineplot(x=avg_growth_by_year.index, y=avg_growth_by_year.values, marker='o')
plt.title("Average Growth by Founding Year")
plt.xlabel("Founding Year")
plt.ylabel("Average Growth (%)")
plt.tight_layout()
plt.show()

 # Industry-Wise Growth
industry_growth = df_clean.groupby('industry')['growth_%'].mean().sort_values(ascending=False).head(15)
print("Top 15 Industries by Average Growth: ", industry_growth)
plt.figure(figsize=(12, 6))
sns.barplot(x=industry_growth.values, y=industry_growth.index, hue=industry_growth.index, palette='mako', legend=False)
plt.title("Top 15 Industries by Average Growth (%)")
plt.xlabel("Average Growth (%)")
plt.ylabel("Industry")
plt.tight_layout()
plt.show()


# Call plotting function
plot_all()

# Features and target
features = ['revenue', 'workers', 'company_age_2019', 'industry']
target = 'growth_%'

# Drop rows with missing target or features
df_model = df_clean[features + [target]].dropna()

# Define X and y
X = df_model[features]
y = df_model[target]

# One-hot encode categorical features
categorical_features = ['industry']
numerical_features = ['revenue', 'workers', 'company_age_2019']

# Column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

# Model pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(random_state=42))
])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
print("Predictive Modeling Results:")
print(f"R² Score: {r2_score(y_test, y_pred):.2f}")
print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred):.2f}")



