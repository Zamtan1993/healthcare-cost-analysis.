Healthcare Billing Analysis & Predictive Modeling

📝 Executive Summary
This project aimed to predict patient billing amounts using demographic and clinical data. After conducting thorough data cleaning, feature engineering, and applying a RandomForestRegressor, the analysis revealed that the provided dataset does not contain the necessary features to predict billing outcomes. This finding serves as a case study in data quality and the importance of feature selection in healthcare analytics.

🛠 Methodology
Data Cleaning: Implemented SimpleImputer to handle missing values and utilized custom logic to convert raw strings into usable numerical formats.

Feature Engineering: * Calculated Length_of_Stay by processing admission and discharge date timestamps.

Created Age_Group bins to capture non-linear impacts of age on medical costs.

Modeling: Utilized a RandomForestRegressor for non-linear pattern recognition.

Evaluation: Used MAE (Mean Absolute Error) and R-squared to measure model performance.

📈 Key Findings
Low Predictive Power: The model yielded an R 
2
  of 0.04, indicating that the chosen features explain very little of the variance in billing.

Correlation Analysis: A Pearson correlation analysis confirmed that demographic variables (Gender, Blood Type, Age) have essentially zero correlation with the final billing amount.

Conclusion: The billing amounts in this dataset likely depend on proprietary clinical fee schedules or surgical procedure codes not present in the features.

🚀 How to Run:

Ensure the required dependencies are installed:

Bash
pip install -r requirements.txt
Run the analysis script:

Bash
python model.py
📦 Requirements
pandas, numpy

scikit-learn

matplotlib, seaborn

<img width="673" height="379" alt="image_2" src="https://github.com/user-attachments/assets/7f1c632f-1d54-43c1-9cf2-e4860fa736b5" />
