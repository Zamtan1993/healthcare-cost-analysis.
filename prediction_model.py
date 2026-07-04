import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error

# 1. Load the dataset
df = pd.read_csv('/Users/mdtanimbinana/Downloads/healthcare_dataset 2.csv')

# 2. Drop columns that are identifiers
# We use errors='ignore' so it doesn't crash if they are already dropped
df = df.drop(['Name', 'Date of Admission', 'Doctor', 'Hospital', 'Discharge Date'], axis=1, errors='ignore')

# 3. Identify categorical columns and encode them
# Using include=['object', 'category'] to be compatible with newer pandas versions
categorical_cols = df.select_dtypes(include=['object', 'category']).columns

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# 4. Define features (X) and target (y)
X = df.drop('Billing Amount', axis=1)
y = df['Billing Amount']

# 5. Split and Train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6. Evaluate
predictions = model.predict(X_test)
print(f"Mean Absolute Error: {mean_absolute_error(y_test, predictions):.2f}")



import matplotlib.pyplot as plt
import pandas as pd

# 1. Get feature importance from the trained model
# Note: Ensure this runs after your 'regressor.fit' step
# ... (Previous code: Data loading, cleaning, and training)

# Train Regressor
regressor = RandomForestRegressor(n_estimators=100, random_state=42)
regressor.fit(X_train, y_train)

# --- ADD THIS NEW BLOCK ---
# 1. Extract feature importance
importances = regressor.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# 2. Print the importance table to the console
print("\nFeature Importance Rankings:")
print(feature_importance_df)

# 3. Plot the importance
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='skyblue')
plt.gca().invert_yaxis() # Put the most important at the top
plt.xlabel('Importance Score')
plt.title('Feature Importance for Billing Amount Prediction')
plt.show()
# --------------------------

# Predict and Evaluate
y_pred = regressor.predict(X_test)
print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
print(f"R-squared: {r2_score(y_test, y_pred):.4f}")

