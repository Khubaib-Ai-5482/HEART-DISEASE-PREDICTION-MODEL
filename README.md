# Heart Disease Prediction using Random Forest

This project builds a **classification system** to predict the presence of heart disease using medical attributes. A **Random Forest Classifier** is trained, evaluated, and interpreted through multiple visualizations.

---

## 📌 Project Objective

To predict whether a patient has **heart disease (1)** or is **healthy (0)** using clinical data, and to understand which features contribute most to the prediction.

This project demonstrates a **complete supervised machine learning workflow**.

---

## 🛠️ Tools & Libraries

* **Python**
* **Pandas** – data handling
* **NumPy** – numerical operations
* **Matplotlib** – plotting
* **Seaborn** – statistical visualization
* **Scikit-learn** – model training and evaluation

---

## 📂 Dataset

**Input File:** `heart.csv`

### Target Variable

* `target`

  * `0` → Healthy
  * `1` → Heart Disease

### Features (examples)

* `age`
* `chol` (cholesterol)
* `thalach` (max heart rate)
* `cp` (chest pain type)
* `trestbps` (resting blood pressure)
* and other clinical indicators

---

## 🔄 Project Workflow

### 1. Data Loading & Inspection

* Dataset shape printed for validation
* Features (`X`) and target (`y`) separated

---

### 2. Exploratory Data Analysis (EDA)

* **Target distribution** to check class balance
* **Age vs Cholesterol** scatter plot by target class
* **Correlation heatmap** to study feature relationships

---

### 3. Train-Test Split

* 80% training data
* 20% testing data
* Fixed random state for reproducibility

---

### 4. Model Training

* Algorithm: **Random Forest Classifier**
* Ensemble-based model for robust classification

---

### 5. Model Evaluation

Metrics used:

* **Accuracy Score**
* **Classification Report** (Precision, Recall, F1-score)
* **Confusion Matrix** (True vs Predicted values)

---

### 6. Model Interpretation

* **Feature Importance** plot to identify most influential medical factors

---

### 7. Error Analysis

* **Residual distribution** to analyze misclassifications

---

## 📈 Visual Outputs

* Target class distribution
* Confusion matrix heatmap
* Feature importance bar chart
* Residuals distribution
* Scatter plots for medical insights
* Feature correlation heatmap

---

## 🚀 How to Run

1. Clone the repository
2. Install dependencies:

   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```
3. Place `heart.csv` in the project directory
4. Run the Python script

---

## 📌 Use Cases

* Medical decision support systems
* Binary classification practice
* Healthcare data analysis
* Portfolio project for ML / AI roles

---

## 👤 Author

**Khubaib**
Aspiring AI Engineer | Machine Learning & Healthcare Analytics

---

## ⭐ Notes

* Random Forest handles non-linearity well
* Feature importance helps model interpretability
* Model performance can be improved using hyperparameter tuning

---

If you find this project useful, feel free to ⭐ the repository and extend it with advanced models like XGBoost or SHAP analysis.
