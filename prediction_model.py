import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Load the dataset
df = pd.read_csv('/Users/mdtanimbinana/Downloads/healthcare_dataset 2.csv')

# 2. Feature Engineering
# Create 'Length_of_Stay'
df['Date of Admission'] = pd.to_datetime(df['Date of Admission'])
df['Discharge Date'] = pd.to_datetime(df['Discharge Date'])
df['Length_of_Stay'] = (df['Discharge Date'] - df['Date of Admission']).dt.days

# Create 'Age_Group'
df['Age_Group'] = pd.cut(df['Age'], bins=[0, 18, 40, 65, 120], labels=[0, 1, 2, 3])

# 3. Clean Data: Drop original identifiers
df = df.drop(['Name', 'Date of Admission', 'Doctor', 'Hospital', 'Discharge Date', 'Room Number', 'Age'], axis=1, errors='ignore')

# 4. Handle Missing Data
df = df.dropna(subset=['Billing Amount'])
imputer = SimpleImputer(strategy='most_frequent')
df_filled = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# 5. Encoding
for col in df_filled.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df_filled[col] = le.fit_transform(df_filled[col].astype(str))

# 6. Define X and y
X = df_filled.drop('Billing Amount', axis=1)
y = df_filled['Billing Amount'].astype(float)

# 7. Split and Train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
regressor = RandomForestRegressor(n_estimators=100, random_state=42)
regressor.fit(X_train, y_train)

# 8. Evaluation
y_pred = regressor.predict(X_test)
print(f"MAE: {mean_absolute_error(y_test, y_pred):.2f}")
print(f"R-squared: {r2_score(y_test, y_pred):.4f}")

# 9. Correlation Analysis
# Convert to numeric first to ensure correlation works
df_numeric = df_filled.apply(pd.to_numeric)
correlation = df_numeric.corr()['Billing Amount'].sort_values(ascending=False)
print("\nCorrelation with Billing Amount:")
print(correlation)

# 10. Visualization
importances = regressor.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances}).sort_values(by='Importance', ascending=True)

plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='salmon')
plt.title('Enhanced Feature Importance')
plt.show()
