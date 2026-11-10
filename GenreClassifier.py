import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# 1. LOAD AND FILTER DATA
df = pd.read_csv('spotify-tracks-dataset-detailed.csv')

features = ['danceability', 'energy', 'valence', 'tempo', 'loudness', 'acousticness', 'instrumentalness', 'speechiness', 'liveness' ]
target = 'track_genre'

top_genres=df[target].value_counts().head(12).index
df_filtered=df[df[target].isin(top_genres)]

X = df_filtered[features]
y = df_filtered[target]

print(f"Filtered dataset size: {len(X)} songs across {len(top_genres)} genres ")
print("Generes included:", list(top_genres))

# 2. Split the data

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42, stratify=y)

# 3.Random forest classiffier

rf_genre_model = RandomForestClassifier(n_estimators=200,n_jobs=-1,random_state=42)
rf_genre_model.fit(X_train, y_train)
print("\nGenre classifier training complete!")

# 4. Modewl evaluation

y_pred = rf_genre_model.predict(X_test)

print("\nClassification report")
print(classification_report(y_test, y_pred))

# 5. calculating the confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=rf_genre_model.classes_)

plt.figure(figsize=(12,10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=rf_genre_model.classes_,
            yticklabels=rf_genre_model.classes_)

plt.title('Confusion matrix: Which genres get mixed up?', fontsize = 14)
plt.ylabel('Actual Genre', fontsize = 12)
plt.xlabel('Actual Genre', fontsize = 12)
plt.xticks(rotation = 45, ha='right')
plt.yticks(rotation = 0)
plt.tight_layout()
plt.show()

# 6. Feature importances

importances = rf_genre_model.feature_importances_

feature_imp_df = pd.DataFrame({
    'Feature': features,
    'Importance': importances
}).sort_values(by='Importance', ascending=True) # Sort ascending for horizontal bars

plt.figure(figsize=(8, 5))
sns.barplot(x='Importance', y='Feature', data=feature_imp_df, palette='viridis')
plt.title('Which Audio Features Define a Genre?')
plt.xlabel('Importance Score')
plt.show()