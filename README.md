# 🏦 Bank Customer Churn Prediction

### 🚀 Advanced Machine Learning Pipeline: From Baseline to Gradient Boosting

---

## 📋 Project Overview
Customer churn (attrition) is a critical metric for banking institutions. It is significantly more expensive to acquire a new customer than to retain an existing one. This project implements a comprehensive Data Science workflow to predict which customers are likely to leave the bank based on their demographic and financial profile.

By analyzing a dataset of 10,000 customers, we identify the key drivers of churn and deploy a high-performance machine learning model to proactively flag at-risk clients.

---

## 🎯 Objectives
* **Data Refinement:** Rigorous cleaning, handling class imbalance (SMOTE), and categorical encoding.
* **Model Benchmarking:** Competitive analysis of 7 different classification algorithms.
* **Hyperparameter Optimization:** Fine-tuning models using `GridSearchCV` and `K-Fold Cross-Validation` to prevent overfitting.
* **Business Insights:** Deriving actionable strategies to reduce churn rates based on feature importance.

---

## 🛠 Tech Stack & Tools

| Category | Libraries & Tools |
| :--- | :--- |
| **Language** | Python 3.x |
| **Data Manipulation** | `pandas`, `numpy` |
| **Visualization** | `seaborn`, `matplotlib` |
| **Preprocessing** | `Scikit-Learn`, `Imbalanced-Learn (SMOTE)` |
| **Classic ML** | Logistic Regression, KNN, SVM, Decision Tree, Random Forest |
| **Ensemble/Boosting** | XGBoost, LightGBM |

---

## ⚙️ The Analysis Workflow

The project follows a structured professional pipeline:

1.  **Data Governance & Cleaning:**
    * Removal of non-predictive unique identifiers (`RowNumber`, `CustomerId`, `Surname`).
    * Rigorous null-value integrity checks.

2.  **Exploratory Data Analysis (EDA):**
    * **Univariate Analysis:** Inspecting distributions of Age, Credit Score, and Balance.
    * **Bivariate Analysis:** correlating categorical variables (Geography, Gender) with Churn.
    * **Multivariate Analysis:** Heatmaps to detect collinearity.

3.  **Feature Engineering & Preprocessing:**
    * **One-Hot Encoding:** Converting categorical variables (`Geography`, `Gender`) into numeric format.
    * **Feature Scaling:** Applying `StandardScaler` to normalize numerical ranges (Credit Score, Balance, Estimated Salary).
    * **Imbalance Handling:** Utilizing **SMOTE (Synthetic Minority Over-sampling Technique)** to balance the dataset (approx. 20% churn rate in raw data).

4.  **Model Benchmarking:**
    * Training and evaluating 7 distinct models:
        1.  Logistic Regression
        2.  K-Nearest Neighbors (KNN)
        3.  Support Vector Machine (SVM)
        4.  Decision Tree
        5.  Random Forest
        6.  XGBoost
        7.  LightGBM

5.  **Hyperparameter Tuning:**
    * Applying `GridSearchCV` to find the optimal parameters for the top-performing models.
    * Validating stability using **K-Fold Cross-Validation**.

---

## 🏆 Key Results & Champion Model

After a rigorous tournament of machine learning models, **LightGBM** was selected as the champion model.

* **Selected Model:** LightGBM (Light Gradient Boosting Machine)
* **Cross-Validation Score:** **90.15%**
* **Stability:** Minimal **3.2% Overfit Gap** between training and test sets.
* **Performance:** The model successfully captures complex non-linear relationships between customer activity, product ownership, and churn probability.

---

## 📊 Business Insights & Recommendations

Based on the model's feature importance analysis, the following strategic actions are recommended:

1.  **Target High-Value/High-Risk Segments:**
    * Analysis reveals higher churn variance in customers with **high balances** and specific **age demographics**. Personalized retention offers should be targeted here.

2.  **Proactive Engagement:**
    * The model allows the bank to identify **9 out of 10 potential churners** accurately before they leave.
    * *Action:* Trigger automated "Health Check" emails or relationship manager calls when churn probability crosses 70%.

3.  **Product Ecosystem:**
    * Customers with active credit cards or specific product bundles show distinct retention patterns. Cross-selling additional sticky products (like mortgages or investment accounts) to single-product users may reduce attrition.

---

## 💻 How to Run This Project

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/ShariqAdnan-03/churn-analysis-project.git](https://github.com/ShariqAdnan-03/churn-analysis-project.git)
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run the Notebook:**
    Open `ChurnAnalysis.ipynb` in Jupyter Notebook, Google Colab, or VS Code and execute the cells sequentially.

---

*Project developed by **Shariq Adnan**.*
