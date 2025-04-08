# Predicting-Student-Success-and-Failure

A machine learning project that predicts university student dropout and academic success rates with 92% accuracy. This project compares multiple algorithms including KNN, Decision Tree, Logistic Regression, SVM, Random Forest, Gaussian Naive Bayes, AdaBoost, and XGBoost.

## Tools & Technologies
- ![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white) Python
- ![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white) Pandas
- ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white) NumPy
- ![Matplotlib](https://img.shields.io/badge/Matplotlib-008C45?style=flat&logo=matplotlib&logoColor=white) Matplotlib
- ![LaTeX](https://img.shields.io/badge/LaTeX-008080?style=flat&logo=latex&logoColor=white) LaTeX
- ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white) scikit-learn
- ![Jupyter Notebook](https://img.shields.io/badge/Jupyter-FFB13B?style=flat&logo=jupyter&logoColor=white) Jupyter Notebook

## Key Features
- Data preprocessing pipeline with SMOTE for handling class imbalance
- Feature selection using Pearson's correlation to identify redundant features
- Hyperparameter tuning with RandomSearchCV for optimal model performance
- Random Forest and SVM classifiers achieve highest accuracy (92%)

## Dataset
Utilizes a dataset of 4,424 records with 37 attributes from UCI Machine Learning Repository, containing academic, demographic and socioeconomic student data.

## Methodology
1. Data preprocessing with SMOTE to balance classes
2. Feature selection to remove highly correlated attributes
3. Feature scaling with StandardScaler
4. Model training and evaluation using multiple algorithms
5. Hyperparameter tuning for top-performing models
   
## Results

| Classifiers           | Accuracy | Precision | Recall  | F1-Score |
|-----------------------|----------|-----------|---------|----------|
| KNN                   | 89%      | 91.67%    | 83.81%  | 88.46%   |
| Decision Tree         | 91%      | 91%       | 90%     | 90%      |
| Logistic Regression   | 91%      | 90.93%    | 90.71%  | 90.82%   |
| SVM                   | 92%      | 93.67%    | 88.09%  | 90.8%    |
| Random Forest         | 92%      | 93.93%    | 88.57%  | 91.17%   |
| Gaussian Naive Bayes  | 87%      | 90.64%    | 80.71%  | 85.39%   |
| XGBoost               | 56%      | 75%       | 58%     | 49%      |
| AdaBoost              | 90%      | 90.49%    | 88.33%  | 89.39%   |

## Conclusion
This project demonstrates the potential for machine learning to identify at-risk students early, enabling timely intervention and support measures to improve student retention rates. Random Forest and SVM models performed best, both achieving 92% accuracy.

## Limitations
- Uneven class distribution can bias models toward majority class
- Dataset from a single institution limits generalizability
- Missing influential factors like psychological metrics and peer influence

## Future Work
- Incorporate additional features like psychological factors and extracurricular activities
- Test models on data from different educational institutions
- Implement an early warning system based on the best-performing models
