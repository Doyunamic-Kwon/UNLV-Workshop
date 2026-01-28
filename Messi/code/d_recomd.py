# Topic-based Platform and Content Type Recommendation System
# Executing code for the 'Education' topic only

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error

# [Step 1] Data Collection
df = pd.read_csv('social_media_viral_content_dataset.csv')

# [Step 2] Data Preparation
# 1. Target Preprocessing: Convert views to numeric and apply Log Transformation 
# (To reduce data variance and ensure training stability)
df['views'] = pd.to_numeric(df['views'], errors='coerce').fillna(0)
df['target'] = np.log1p(df['views'])

# 2. Feature Selection: Topic, Platform, Content Type
X_raw = df[['topic', 'platform', 'content_type']]

# 3. One-Hot Encoding: Convert categorical data into numerical data
X = pd.get_dummies(X_raw, dtype=int)
y = df['target']

# 4. Data Splitting: Split into Training (80%) and Testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# [Step 3] Model Selection
rf = RandomForestRegressor(random_state=42)

# [Step 4] Model Training & [Step 6] Hyperparameter Tuning
# Find the optimal tree depth and number of estimators using GridSearchCV
param_grid = {'n_estimators': [100, 200], 'max_depth': [10, 20, None]}
grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='r2', n_jobs=-1)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

# [Step 5] Evaluation
y_pred = best_model.predict(X_test)
r2 = r2_score(y_test, y_pred)
# Inverse log transformation (expm1) to calculate MAE in actual view units
mae = mean_absolute_error(np.expm1(y_test), np.expm1(y_pred))

print(f"--- Model Performance Evaluation ---")
print(f"R2 Score: {r2:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.2f} views")

# [Step 7] Results Interpretation & Visualization (Recommendation Engine)
def get_recommendation(target_topic):
    # 1. Create all possible combinations of platform x content_type for the target topic
    platforms = df['platform'].unique()
    c_types = df['content_type'].unique()
    comb = pd.DataFrame([{'topic': target_topic, 'platform': p, 'content_type': c} 
                         for p in platforms for c in c_types])
    
    # 2. Encode data and align columns to match the trained model's format
    X_comb = pd.get_dummies(comb, dtype=int).reindex(columns=X.columns, fill_value=0)
    
    # 3. Predict views and perform inverse log transformation
    comb['predicted_views'] = np.expm1(best_model.predict(X_comb))
    comb['Strategy'] = comb['platform'] + " @ " + comb['content_type']
    
    return comb.sort_values(by='predicted_views', ascending=False)

# Get recommendations for 'Education' as an example
edu_rec = get_recommendation('Education')

# Visualization
plt.figure(figsize=(12, 6))
sns.barplot(data=edu_rec.head(10), x='predicted_views', y='Strategy', palette='viridis')
plt.title(f"Top 10 Recommended Strategies for 'Education'", fontsize=14, fontweight='bold')
plt.xlabel("Predicted Average Views")
plt.tight_layout()
plt.show()

print("\nTop Recommendations for 'Education'")
print(edu_rec[['Strategy', 'predicted_views']].head(3))