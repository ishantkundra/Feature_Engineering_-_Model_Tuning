# 🧪 FMT Project – Semiconductor Yield Classification with Feature Engineering & Model Tuning

## 📌 Project Overview

This project tackles a real-world use case from the **semiconductor manufacturing domain**, where hundreds of sensor signals (features) are collected to predict whether a unit **passes or fails** internal yield tests. The objective is to preprocess high-dimensional data, reduce noise, and build the most efficient and accurate classifier using advanced **supervised learning techniques** and **model tuning strategies**.

---

## 🧬 Domain: Semiconductor Manufacturing

### 🧠 Objective:
Build a classifier to predict **Pass/Fail** yield outcomes using over **590 sensor signals** (features), while identifying the most relevant signals and optimizing model performance via hyperparameter tuning and dimensionality reduction.

---

## 📊 Dataset

- `signal-data.csv` – (1567 rows × 592 columns)  
  Each row represents a manufacturing instance with 591 sensor readings (features) and 1 target column (`-1` for pass, `1` for fail).

---

## ⚙️ Key Steps & Workflow

1. **Data Understanding & Cleaning**
   - Loaded dataset and checked shape, data types, and summary
   - Removed features with >20% missing values
   - Imputed remaining nulls with column mean
   - Dropped constant-valued features and highly correlated features

2. **Feature Engineering**
   - Applied **Principal Component Analysis (PCA)** for dimensionality reduction
   - Ensured sufficient variance retention (≥90%)

3. **Preprocessing**
   - Separated predictors and target
   - Handled class imbalance if present
   - Standardized features and performed train-test split
   - Verified statistical distribution similarity across splits

4. **Model Training & Tuning**
   - Trained multiple classifiers (Logistic Regression, SVM, etc.)
   - Applied **GridSearchCV** for hyperparameter tuning
   - Used cross-validation for robust performance estimation

5. **Evaluation**
   - Compared performance across models using:
     - Accuracy
     - Precision, Recall, F1-Score
     - Confusion Matrix
   - Selected best model and saved with Pickle

6. **Conclusion**
   - Summarized learnings and model choice rationale

---

## 🛠️ Tools, Libraries & Skills Used

- **Languages & Tools:** Python, Jupyter Notebook, HTML Export
- **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
- **Skills:**
  - Supervised Learning
  - Feature Engineering
  - PCA (Principal Component Analysis)
  - Model Evaluation & Cross-validation
  - Grid Search Hyperparameter Tuning
  - Classification Metrics
  - Pickling Models for Reuse

---

## 📁 Repository Structure

<pre>

.
├── code/
│   ├── FMT_IK_.ipynb                          # Main project notebook
│   └── FMT_IK_.html                           # Exported HTML version
│
├── dataset/
│   └── signal-data.csv                        # High-dimensional sensor data
│
├── Problem Statement/
│   └── FMT - Problem_Statement.pdf            # Detailed instructions & context
│
├── .gitignore
└── README.md                                  # This file

</pre>

---

## 💡 Key Learnings

- How to manage and clean high-dimensional sensor data  
- Selecting meaningful features through PCA and statistical reasoning  
- Building robust classification models in imbalanced and noisy environments  
- Improving model performance with Grid Search and cross-validation  
- Developing a complete ML pipeline from raw data to deployment-ready model

---

## ✍️ Author

**Ishant Kundra**  
📧 [ishantkundra9@gmail.com](mailto:ishantkundra9@gmail.com)  
🎓 Master’s in Computer Science | AIML Track
