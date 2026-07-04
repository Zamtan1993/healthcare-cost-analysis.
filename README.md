Healthcare Billing Cost Prediction

🏥 Project Overview
This project performs an end-to-end machine learning analysis on a healthcare dataset. The goal was to build a predictive model to estimate hospital Billing Amount based on patient demographics, medical conditions, and admission details, and to identify which factors most significantly influence these costs.

📊 Methodology
Data Cleaning: Removed non-predictive identifiers (Name, Doctor, Hospital) and ensured data integrity.

Feature Engineering: Calculated Length of Stay (the number of days between admission and discharge) as a potential predictor for hospital costs.

Data Preprocessing: Utilized LabelEncoder to convert categorical variables (such as "Medical Condition" and "Insurance Provider") into a numerical format suitable for machine learning algorithms.

Modeling: Implemented a RandomForestRegressor to predict hospital billing amounts, utilizing a 80/20 train-test split.

📈 Key Insights
Prediction Performance: The model achieved a Mean Absolute Error (MAE) of approximately $11,664.

Feature Importance: Analysis revealed that while the model utilized various patient demographics, the available features showed a weak correlation with total billing. This suggests that actual healthcare costs in this dataset are likely driven by clinical procedures or illness severity factors not fully captured in the current variables.

🛠️ Tech Stack
Language: Python

Core Libraries: pandas, scikit-learn, matplotlib, numpy

Project structure:
├── data/
│   └── healthcare_dataset 2.csv
├── images/
│   └── feature_importance.png
├── src/
│   └── prediction_model.py
├── requirements.txt
└── README.md

🚀 How to Run
Clone this repository to your local machine.

Install the required dependencies:

pip install -r requirements.txt

Ensure your dataset is located in the data/ folder.

Run the analysis script:
python src/prediction_model.py

🔮 Future Improvements
Advanced Modeling: Explore Gradient Boosting frameworks like XGBoost or LightGBM to potentially improve regression performance.

Data Enrichment: Incorporate external datasets containing CPT/ICD-10 procedure codes to better reflect real-world clinical billing structures.
