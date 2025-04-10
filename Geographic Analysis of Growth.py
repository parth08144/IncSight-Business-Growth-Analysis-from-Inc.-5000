import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# -------------------------------
# 3. Geographic Analysis of Growth
# -------------------------------

# Average growth by state
state_growth = df_clean.groupby('state')['growth_%'].mean().sort_values(ascending=False).head(10)
print("Top 10 States by Average Growth: ", state_growth)

# Companies per state
companies_per_state = df_clean['state'].value_counts().head(10)
print("Companies per State: ", companies_per_state)
