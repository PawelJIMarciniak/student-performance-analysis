# student-performance-analysis

# Student Performance Analysis & Grade Prediction (G3)

This project focuses on data analysis and building machine learning models (classification, regression, and clustering) to predict students' final grades (`G3`) based on demographic, social, and school-related features. The analysis is performed on a dataset of 649 students from a Portuguese secondary school.

## 📋 Project Objectives & Constraints

The main goal is to accurately predict the final grade (`G3`) while adhering to strict data science guidelines:
* **Feature Constraints:** The predictors `G1` and `G2` (interim grades) **were completely removed** to ensure the models rely purely on independent background attributes, preventing data leakage from previous periods.
* **Data Splitting:** The dataset was split into training (80%) and testing (20%) sets using a fixed random seed to ensure strict isolation of the test data during Exploratory Data Analysis (EDA) and feature engineering.

---

## 🛠️ Tech Stack & Libraries

The workflow is implemented in Python within a Jupyter Notebook environment using the following ecosystem:
* **Data Manipulation:** `pandas`, `numpy`
* **Data Visualization:** `matplotlib`, `seaborn`
* **Machine Learning:** `scikit-learn` (LinearRegression, KMeans, StandardScaler, LabelEncoder)
* **Advanced Gradient Boosting:** `xgboost` (XGBClassifier)
* **Statistical Analysis & Diagnostics:** `scipy`, `statsmodels`

---

## 📈 Methodology & Project Stages

### 1. Data Cleaning & Preprocessing
* Checked for missing values and validated the integrity of numeric ranges.
* Manually mapped binary variables (0/1) to maintain full control over feature coefficient interpretation.
* Encoded categorical variables using One-Hot Encoding, ensuring feature synchronization between the training and test sets.

### 2. Exploratory Data Analysis (EDA)
* Conducted exclusively on the training set to prevent information leakage.
* Analyzed the distribution of the target variable (`G3`) and generated feature correlation matrices.
* Identified key predictors with the highest statistical impact on student performance (e.g., previous failures, mother's education, desire for higher education, and weekly study time).

### 3. Classification Model (G3 as a Categorical Variable)
* Built an **XGBoost Classifier** to predict the exact grade class.
* Evaluated model performance on the unseen test set using classification metrics and a confusion matrix.

### 4. Estimation Model (G3 as a Continuous Variable)
* Built a **Linear Regression (OLS)** model using standardized features (`StandardScaler`).
* Predicted continuous values were rounded to the nearest integer for evaluation.
* **Model Diagnostics:** Rigorously tested linear regression assumptions, including the Shapiro-Wilk normality test on residuals, Q-Q plots, homoscedasticity evaluation, Durbin-Watson autocorrelation test, and Cook's Distance to identify influential outliers.

### 5. Student Segmentation (Clustering)
* Selected key educational and social predictors to group students using the **K-Means** algorithm.
* Determined the optimal number of clusters ($k=3$) using the **Elbow Method** (Inertia).
* Profiled each cluster on the training set (e.g., identifying risk groups vs. high achievers) and applied the schema to the test set to validate the relationship between cluster assignment and final `G3` grades.

---

## 📊 Evaluation Metrics

Both the classification and regression models were evaluated on the test set using three robust metrics:
1. **Accuracy:** The percentage of perfectly predicted grades.
2. **Accuracy within a $\pm1$ margin:** The percentage of predictions that were exact or off by at most 1 grade point.
3. **Mean Absolute Error (MAE):** The average absolute distance between the predicted and actual grades.
