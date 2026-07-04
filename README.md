Healthcare Billing Analysis & Predictive Modeling

📝 Executive Summary: This project evaluates the relationship between patient demographics, clinical variables, and total billing amounts using machine learning. After conducting thorough data cleaning, feature engineering, and applying a RandomForestRegressor, the analysis revealed that the provided dataset lacks the predictive signal required to accurately forecast billing outcomes.This result serves as a valuable case study in data quality and highlights the critical importance of feature selection and domain-specific inputs in healthcare analytics.

Methodology:

The project follows a rigorous end-to-end data science pipeline:Data Cleaning: Utilized SimpleImputer to address missing values and custom logic to normalize categorical and temporal data.Feature Engineering:Derived Length_of_Stay by calculating the delta between admission and discharge timestamps.Binning Age into Age_Group cohorts to better capture non-linear impacts on medical costs.Modeling: Deployed a RandomForestRegressor for non-linear pattern recognition.Evaluation: Employed MAE (Mean Absolute Error) and $R^2$ (Coefficient of Determination) to benchmark performance, complemented by Pearson correlation analysis to quantify feature influence.

📈 Key Findings: 
Low Predictive Power: The model achieved an $R^2$ of 0.04, indicating that the input features explain very little variance in billing amounts.
Statistical Independence: A Pearson correlation analysis confirmed that demographic variables (Gender, Blood Type, Age) have essentially zero correlation with the final billing amount.
Domain Conclusion: The billing structure in this dataset appears independent of patient demographics. Actual costs are probably governed by proprietary fee schedules or specific medical procedure codes that were not available in this dataset.

Visualizing Feature Relationships. To validate these findings, I implemented a correlation heatmap to analyze feature interactions: Python import seaborn as sns
plt.figure(figsize=(10, 8))
sns.heatmap(df_numeric.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Analysis of Healthcare Features')
plt.show()

🚀 Getting Started Prerequisites: Ensure you have the required dependencies installed: Bashpip install -r requirements.txt
Execution: Run the analysis script to process the data and generate the performance reports: Bash python model.py
📦 Requirements: pandas, numpy, scikit-learn, matplotlib, seaborn

<img width="673" height="379" alt="image_2" src="https://github.com/user-attachments/assets/7f1c632f-1d54-43c1-9cf2-e4860fa736b5" />
