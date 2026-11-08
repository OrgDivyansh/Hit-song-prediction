import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the dataeset
df = pd.read_csv('spotify-tracks-dataset-detailed.csv')

features = ['danceability', 'energy', 'valence', 'tempo', 'loudness']
df['is_hit']= (df['popularity']>70).astype(int)
target = 'is_hit'

#  2 Create X features and y target
X= df[features]
y= df[target]

# Data cleaning
data_clean= df[features +[target]].dropna()

X = data_clean[features]
y = data_clean[target]

print(f"\nTotal songs: {len(y)}")
print(f"Number of Hits (1): {y.sum()} ({(y.sum()/len(y)*100):.2f}%)")
print(f"Number of flops (0): {len(y) - y.sum()} ({((len(y) - y.sum())/len(y)*100):.2f}%)")
from sklearn.model_selection import train_test_split

#  3. Split Data 80% for training and 20% for resting
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42, stratify=y )

print(f"\nTraining set Hits: {y_train.sum()}")
print(f"Testing set size {y_test.sum}")

from sklearn.ensemble import RandomForestClassifier

# 4. Initialise the model

rf_model = RandomForestClassifier(n_estimators=100, random_state=42,n_jobs=-1,class_weight='balanced')

rf_model.fit(X_train,y_train)
print("\nClassification Model training complete!")

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# 4. Make predictions on the test set
y_pred = rf_model.predict(X_test)

# calculate
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1= f1_score(y_test, y_pred)

print("\n--- Model Evaluation ---")
print(f"Accuracy:  {accuracy:.4f} (Overall correctness)")
print(f"Precision: {precision:.4f} (When it predicts a Hit, how often is it right?)")
print(f"Recall:    {recall:.4f} (Out of all actual Hits, how many did it find?)")
print(f"F1-Score:  {f1:.4f} (Balance between Precision and Recall)\n")

print("--- Detailed Classification Report ---")
print(classification_report(y_test, y_pred, target_names=['Flop (0)', 'Hit (1)']))


#Plotting the confusion matrix
cm = confusion_matrix(y_test,y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d' , cmap='Blues',
            xticklabels=['Predicted Flop','Predicted Hit'],
            yticklabels=['Actual Flop, Actual Hit'])
plt.title('Confusion Matrix')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.show()