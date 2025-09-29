#  Breast Cancer Classification (Machine Learning)

An end-to-end **machine learning project** that predicts whether a breast tumor is **Benign (0)** or **Malignant (1)**.  
Includes **data preprocessing, exploratory data analysis (EDA), feature engineering, model training, and evaluation** using Logistic Regression, Random Forest, and SVM.

---

Breast cancer is one of the most common cancers worldwide. Early detection can significantly improve survival rates.  
This project applies ML techniques to classify tumors and assist in early diagnosis.

---

##  Dataset
- **Source**: [Breast Cancer Wisconsin (Diagnostic) Dataset]([https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data))  
- **Samples**: 569  
- **Features**: 30 numeric features (e.g., `radius_mean`, `texture_mean`)  
- **Target**: Diagnosis → `0` = Benign, `1` = Malignant  

---

##  Exploratory Data Analysis
- Benign vs Malignant distribution (~60% vs ~40%)  
- Strong feature correlations (`radius_mean`, `perimeter_mean`)  
- Visualizations: histograms, boxplots, heatmaps  

---

##  Feature Engineering
- Dropped irrelevant columns (`id`, `Unnamed: 32`)  
- Standardized features with **StandardScaler**  
- Removed highly correlated features (|r| > 0.90)  
- Selected top 10 features via **ANOVA F-test**  
- Applied **PCA** (95%+ variance explained)  

---

##  Models Used
- Logistic Regression (baseline)  
- Random Forest Classifier  
- Support Vector Machine (RBF kernel)  

---

##  Results

| Model               | Accuracy | Precision | Recall | F1 Score |
|----------------------|----------|-----------|--------|----------|
| Logistic Regression | 96.5%    | 95.8%     | 96.2%  | 96.0%    |
| Random Forest       | 98.2%    | 97.9%     | 98.0%  | 98.0%    |
| SVM (RBF Kernel)    | 97.5%    | 97.0%     | 97.3%  | 97.1%    |

Random Forest performed best overall.

---
