# Identify features that impact 'views' the most without a machine learning model (No One-Hot Encoding).
# Conclusion: Impact rank is Topic > Content Type > Platform.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# [Step 1] Data Loading and Cleaning
df = pd.read_csv('social_media_viral_content_dataset.csv')
df['views'] = pd.to_numeric(df['views'], errors='coerce').fillna(0)

# [Step 2] Calculate Influence per Feature (Check volatility of average views)
features = ['platform', 'content_type', 'topic']
influence_scores = {}

for col in features:
    # Calculate the mean views for each category, then compute the standard deviation of those means.
    # A higher standard deviation indicates the feature has a greater impact on views.
    mean_views = df.groupby(col)['views'].mean()
    influence_scores[col] = mean_views.std()

# Create a result DataFrame
impact_df = pd.DataFrame({
    'Feature': influence_scores.keys(),
    'Influence_Score': influence_scores.values()
}).sort_values(by='Influence_Score', ascending=False)

# [Step 3] Visualization
plt.figure(figsize=(15, 6))

# Plot: Ranking of influence by feature
sns.barplot(data=impact_df, x='Feature', y='Influence_Score', palette='magma')
plt.title('Which Feature Impacts Views the Most?', fontsize=14, fontweight='bold')
plt.ylabel('Influence Score (Std of Mean Views)')

# Identify the top feature for the final conclusion
top_feature = impact_df.iloc[0]['Feature']
plt.show()

# [Step 4] Final Conclusion Output
print(f"Analysis Result: The feature with the greatest impact on views is '{top_feature}'.")