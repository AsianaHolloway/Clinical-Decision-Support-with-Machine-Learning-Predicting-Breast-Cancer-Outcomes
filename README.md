# 🌸 Clinical Decision Support with Machine Learning: Predicting Breast Cancer Outcomes  

![Status](https://img.shields.io/badge/Status-Completed-success?style=flat-square)  
![Python](https://img.shields.io/badge/Python-3.9-blue?style=flat-square&logo=python)  
![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-orange?style=flat-square&logo=scikitlearn)  
![Google Colab](https://img.shields.io/badge/Platform-Google%20Colab-yellow?style=flat-square&logo=googlecolab)  
![CDSS](https://img.shields.io/badge/Clinical%20Decision%20Support-AI%20Modeling-purple?style=flat-square)

---

## ⭐ Project Overview
This project demonstrates how supervised machine learning can support Clinical Decision Support Systems (CDSS) by predicting breast cancer outcomes (benign vs. malignant) using diagnostic imaging features from the Wisconsin Breast Cancer Dataset.

## 🎥 Final Project Video Presentation

[![Watch the Video](https://github.com/AsianaHolloway/Clinical-Decision-Support-with-Machine-Learning-Predicting-Breast-Cancer-Outcomes/blob/main/results/video_thumbnail.png?raw=true)](https://drive.google.com/file/d/1nNST71wl7-wnZE9ChMDSTAQO7_nKuvpK/view?usp=sharing)


The project includes:

- Data preprocessing and cleaning  
- Handling class imbalance using SMOTE  
- Training four supervised ML models  
- Evaluating model performance (accuracy, precision, sensitivity, AUC)  
- Generating clinical visualizations including ROC curves and confusion matrices  
- Feature importance analysis (Random Forest)  
- A reproducible Google Colab notebook  
- Final slide presentation summarizing findings  

---

## 🎯 Purpose of the Project
The purpose of this project is to build a clinically meaningful machine learning pipeline capable of:

- Predicting tumor malignancy  
- Supporting clinicians with interpretable predictive insights  
- Demonstrating how AI can improve early detection  
- Comparing performance across multiple ML algorithms  
- Showing how predictive modeling integrates into CDSS workflows  

This project aligns with clinical decision-making principles and enhances understanding of AI's role in diagnostic support.

---

## 🧬 Dataset Description
**Dataset:** Wisconsin Diagnostic Breast Cancer  
**Source:** https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data  
**Rows:** 569  
**Features:** 30 diagnostic imaging measurements  
**Target:**  
- 0 = Benign  
- 1 = Malignant  

### ✔ Why This Dataset Is “Good Data”
- Contains no missing values  
- Clinically validated and widely used  
- Captures meaningful tumor imaging characteristics  
- Balanced enough for ML but improved with SMOTE  
- Follows clear labeling conventions  

---

## 🧠 Key Code Documentation

### 📌 Loading & Encoding Data
```python
df = pd.read_csv('/content/breast_cancer_wisconsin.csv')
df['target'] = (df['diagnosis'] == 'M').astype(int)
X = df.drop(columns=['diagnosis', 'target'])
y = df['target']
```
## 📌 Train/Test Split
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)
```
## 📌 Handling Class Imbalance (SMOTE)
```python
sm = SMOTE(random_state=42)
X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)
```
## 📌 Model Dictionary
```python
models = {
    "LogReg": Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(max_iter=500, solver='liblinear'))
    ]),
    "DecisionTree": DecisionTreeClassifier(random_state=42),
    "RandomForest": RandomForestClassifier(n_estimators=300, random_state=42),
    "SVM": Pipeline([
        ('scaler', StandardScaler()),
        ('clf', SVC(kernel='rbf', probability=True, random_state=42))
    ])
}
```
## 📊 Model Performance Summary
🔵 ROC Curves

Support Vector Machine ROC Curve
![ROC SVM](https://github.com/AsianaHolloway/Clinical-Decision-Support-with-Machine-Learning-Predicting-Breast-Cancer-Outcomes/blob/main/results/roc_SVM.png?raw=true)


Random Forest ROC Curve
![ROC RandomForest](https://github.com/AsianaHolloway/Clinical-Decision-Support-with-Machine-Learning-Predicting-Breast-Cancer-Outcomes/blob/main/results/roc_RandomForest.png?raw=true)


Logistic Regression ROC Curve
![ROC LogReg](https://github.com/AsianaHolloway/Clinical-Decision-Support-with-Machine-Learning-Predicting-Breast-Cancer-Outcomes/blob/main/results/roc_LogReg.png?raw=true)


Decision Tree ROC Curve
![ROC DecisionTree](https://github.com/AsianaHolloway/Clinical-Decision-Support-with-Machine-Learning-Predicting-Breast-Cancer-Outcomes/blob/main/results/roc_DecisionTree.png?raw=true)

## 🔴 Confusion Matrices

SVM Confusion Matrix
![Confusion Matrix SVM](https://github.com/AsianaHolloway/Clinical-Decision-Support-with-Machine-Learning-Predicting-Breast-Cancer-Outcomes/blob/main/results/confusion_SVM.png?raw=true)


Random Forest Confusion Matrix
![Confusion Matrix RF](https://github.com/AsianaHolloway/Clinical-Decision-Support-with-Machine-Learning-Predicting-Breast-Cancer-Outcomes/blob/main/results/confusion_RandomForest.png?raw=true)


Logistic Regression Confusion Matrix
![Confusion Matrix LogReg](https://github.com/AsianaHolloway/Clinical-Decision-Support-with-Machine-Learning-Predicting-Breast-Cancer-Outcomes/blob/main/results/confusion_LogReg.png?raw=true)



Decision Tree Confusion Matrix
![Confusion Matrix DT](https://github.com/AsianaHolloway/Clinical-Decision-Support-with-Machine-Learning-Predicting-Breast-Cancer-Outcomes/blob/main/results/confusion_DecisionTree.png?raw=true)


## 🌲 Feature Importance
![Feature Importance RF](https://github.com/AsianaHolloway/Clinical-Decision-Support-with-Machine-Learning-Predicting-Breast-Cancer-Outcomes/blob/main/results/feature_importance_rf.png?raw=true)


## 🩺 Clinical Interpretation

Random Forest and SVM are the best-performing models, achieving AUC values between 0.99 and 1.00.

False-negative rates were extremely low, which is essential in cancer detection.

Key predictive features include:

`perimeter_worst`

`area_worst`

`concave points_mean`

`radius_worst` 

These models can be integrated into CDSS workflows to help clinicians flag potentially malignant tumors earlier.

## 💻 Reproducibility Instructions
▶ Run on Google Colab

Upload breast_cancer_wisconsin.csv

Open the notebook in the /notebooks folder

Click Runtime → Run all

All plots and metrics will be generated automatically

## ▶ Local Setup
```python
pip install -r requirements.txt
```
## 📂 Repository Structure
.
├── README.md
├── notebooks/
│   └── Breast_Cancer_CDSS.ipynb
├── data/
│   └── breast_cancer_wisconsin.csv
├── results/
│   ├── confusion_LogReg.png
│   ├── confusion_SVM.png
│   ├── confusion_RandomForest.png
│   ├── confusion_DecisionTree.png
│   ├── roc_LogReg.png
│   ├── roc_SVM.png
│   ├── roc_RandomForest.png
│   ├── roc_DecisionTree.png
│   ├── metrics_summary.csv
│   ├── cv_summary.csv
│   └── feature_importance_rf.png
└── slides/
    └── CDSS_BreastCancer_Presentation.pdf

## 📚 References

Abdel-Zaher, A. M., & Eldeib, A. M. (2016). Breast cancer classification using deep belief networks. Expert Systems with Applications.

Chaurasia, V., & Pal, S. (2017). Data mining techniques to predict and resolve breast cancer survivability. IJCSMC.

Jović, A., Brkić, K., & Bogunović, N. (2015). Feature selection methods with applications to bioinformatics. MIPRO.

Rajendran, K., Jayabalan, M., & Thiruchelvam, V. (2020). Predicting breast cancer via supervised ML. IJACSA.
