# Final Topic-based Platform and Content Type Recommendation System
# Recommends the Top 5 combinations based on user-inputted topics

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error

# Ignore warning messages (Prevent FutureWarnings)
warnings.filterwarnings('ignore')

# [Step 1] Data Collection
df = pd.read_csv('social_media_viral_content_dataset.csv')

# [Step 2] Data Preparation
# 1. Target Preprocessing: Convert views to numeric and apply Log Transformation
df['views'] = pd.to_numeric(df['views'], errors='coerce').fillna(0)
df['target'] = np.log1p(df['views'])

# 2. Feature Selection and Encoding
X_raw = df[['topic', 'platform', 'content_type']]
X = pd.get_dummies(X_raw, dtype=int)
y = df['target']

# 3. Data Splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# [Steps 3-6] Model Building, Training, and Optimization
# Create the optimal model using RandomForest and GridSearchCV
print("Model training and optimization in progress... Please wait.")
rf = RandomForestRegressor(random_state=42)
param_grid = {'n_estimators': [100, 200], 'max_depth': [10, 20]}
grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='r2', n_jobs=-1)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

# Print Model Evaluation Results
y_pred = best_model.predict(X_test)
print(f"\nModel Ready!")
print(f"Model R2 Score: {r2_score(y_test, y_pred):.4f}")
print(f"Mean Absolute Error (MAE): {mean_absolute_error(np.expm1(y_test), np.expm1(y_pred)):.0f} views")

# [Step 7] Result Interpretation and Recommendation System Function
def run_recommendation_system():
    # 1. Display available topics
    available_topics = sorted(df['topic'].unique())
    print("\n" + "="*50)
    print("List of Available Topics for Analysis")
    print("-" * 50)
    for i, topic in enumerate(available_topics):
        print(f"{i+1}. {topic}")
    print("="*50)

    # 2. Receive topic input from the user
    user_input = input("\nEnter the topic you want recommendations for: ").strip()

    if user_input not in available_topics:
        print(f"'{user_input}' is not in the list. Please enter the exact name.")
        return

    # 3. Generate all Platform x Content Type combinations for the topic and predict
    platforms = df['platform'].unique()
    c_types = df['content_type'].unique()
    
    comb_list = []
    for p in platforms:
        for c in c_types:
            comb_list.append({'topic': user_input, 'platform': p, 'content_type': c})
    
    test_df = pd.DataFrame(comb_list)
    X_test_comb = pd.get_dummies(test_df, dtype=int).reindex(columns=X.columns, fill_value=0)
    
    # 4. Prediction and Result Organization
    test_df['predicted_views'] = np.expm1(best_model.predict(X_test_comb))
    test_df['Strategy'] = test_df['platform'] + " @ " + test_df['content_type']
    results = test_df.sort_values(by='predicted_views', ascending=False).head(5)

    # 5. Result Visualization
    plt.figure(figsize=(10, 6))
    sns.barplot(data=results, x='predicted_views', y='Strategy', hue='Strategy', palette='flare', legend=False)
    plt.title(f"Best Viral Strategies for '{user_input}'", fontsize=15, fontweight='bold')
    plt.xlabel("Predicted Average Views")
    plt.tight_layout()
    plt.show()

    # 6. Text Result Output
    print(f"\nTop 3 Viral Strategies for '{user_input}'")
    print("-" * 50)
    for i, (idx, row) in enumerate(results.head(3).iterrows()):
        print(f"Rank {i+1}: {row['Strategy']} (Est. Views: {int(row['predicted_views']):,} views)")

# Execute system
run_recommendation_system()