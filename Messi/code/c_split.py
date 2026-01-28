# Identify features that impact 'views' the most without a machine learning model (Using One-Hot Encoding).
# Conclusion: Regardless of One-Hot Encoding, 
# the impact rank consistently remains Topic > Content Type > Platform.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Data Loading and Preprocessing
df = pd.read_csv('social_media_viral_content_dataset.csv')
df['views'] = pd.to_numeric(df['views'], errors='coerce').fillna(0)

# 2. One-Hot Encoding (Convert all categories in platform, content_type, and topic into 0 and 1)
features = ['platform', 'content_type', 'topic']
df_encoded = pd.get_dummies(df[features], dtype=int)
df_encoded['views'] = df['views']

# 3. Calculate Correlation between individual items and 'views'
# A higher absolute value indicates a stronger impact (linear relationship) on views.
correlations = df_encoded.corr()['views'].drop('views').abs().sort_values(ascending=False)

# 4. Visualization (Top 15 Impacting Items)
plt.figure(figsize=(12, 8))
sns.barplot(x=correlations.values[:15], y=correlations.index[:15], palette='coolwarm')
plt.title('Top 15 Items Impacting Views (One-Hot Encoded)', fontsize=15, fontweight='bold')
plt.xlabel('Absolute Correlation with Views')
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# 5. Final Conclusion Output
print("[Top 5 Items by Impact on Views]")
print(correlations.head(5))