import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Calculate correlation
correlation = df_clean[['workers', 'revenue']].corr()
print("Correlation between Workforce and Revenue: ", correlation)