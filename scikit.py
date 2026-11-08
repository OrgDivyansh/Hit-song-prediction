import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the dataeset
df = pd.read_csv('spotify-tracks-dataset-detailed.csv')

features = ['danceability', 'energy', 'valence', 'tempo', 'loudness']
target = 'popularity'

#  2 Create X features and y target
X= df[features]
y= df[target]

# Data cleaning
data_clean= df[features +[target]].dropna()

X = data_clean[features]
y = data_clean[target]

print(f"\nDataset size after cleaning {X.shape[0]} rows")

from sklearn.model_selection import train_test_split

#  3. Split Data 80% for training and 20% for resting
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42 )

print(f"Training set size {X_train.shape[0]} songs")
print(f"Testing set size {X_test.shape[0]} songs")

from sklearn.ensemble import RandomForestRegressor

# 4. Initialise the model

rf_model = RandomForestRegressor(n_estimators=100, random_state=42,n_jobs=-1)

rf_model.fit(X_train,y_train)
print("\nModel training complete!")

from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

# 4. Make predictions on the test set
y_pred = rf_model.predict(X_test)

# calculate
mae = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n--- Model evaluation ---")
print(f"Mean absolute error(MAE): {mae:.2f}")
print(f"Root mean squared error(RMSE) {rmse:.2f}")
print(f"R-squared (R2): {r2:.4f}")

# 7. Get feature importances
importances = rf_model.feature_importances_

#Dataframe feature for visulaization
feature_imp_df = pd.DataFrame({
    'Feature': features,
    'Importance': importances}).sort_values(by= 'Importance', ascending =False)

#Plotting the details
plt.figure(figsize=(8,5))
sns.barplot(x='Importance', y='Feature', data=feature_imp_df, palette='viridis')
plt.title('Which audio features predict popularity best?')
plt.xlabel('Importance score')
plt.ylabel('Audio Feature')
plt.show()