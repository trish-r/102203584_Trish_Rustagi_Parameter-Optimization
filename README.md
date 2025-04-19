# 102203584_Trish_Rustagi_Parameter-Optimization
# SVM Optimization on UCI Air Quality Dataset (9358 instances and 15 features)

## Objectives
To evaluate the performance of an optimized Support Vector Machine (SVM) classifier on the **Air Quality dataset** from the UCI repository using random hyperparameter tuning across multiple training samples.

---

## Dataset Description

- **Source**: UCI Machine Learning Repository  
- **Dataset Name**: Air Quality Data Set  
- **UCI ID**: 360  
- **Instances**: 9358
- **Features**: Time-series measurements of air pollutants and environmental variables - 15 Features
- **Target**: `CO(GT)` (converted into 3 categorical bins: Low, Medium, High for multi-class classification)

---

## Methodology

1. **Preprocessing**
   - Dropped missing values
   - Selected relevant numerical features
   - Discretized `CO(GT)` into 3 classes using `pd.qcut()`

2. **Sampling**
   - Performed **10 different 70-30 train-test splits** to generate samples (S1 to S10)

3. **Model**
   - Used `SVC` (Support Vector Classifier) from scikit-learn
   - For each sample, ran **100 random hyperparameter combinations** of:
     - `kernel`: `'linear'`, `'rbf'`, `'poly'`, `'sigmoid'`
     - `C`: `0.1`, `1`, `10`
     - `gamma`: `'scale'`, `'auto'`
   - Recorded the **best accuracy and hyperparameters** per sample

4. **Result Visualization**
   - Plotted **convergence graph** of the best performing sample based on accuracy

---

## Table 1: Comparative performance of Optimized-SVM with different samples

| Sample | Best Accuracy | Best SVM Parameters (Kernel, C, Gamma) |
|--------|---------------|-----------------------------------------|
| S1     | 87.89%        | ('rbf', 10, 'scale')                    |
| S2     | 88.36%        | ('rbf', 1, 'auto')                      |
| S3     | 88.02%        | ('rbf', 10, 'scale')                    |
| S4     | 87.76%        | ('rbf', 10, 'scale')                    |
| S5     | 88.71%        | ('rbf', 10, 'auto')                     |
| S6     | 88.28%        | ('rbf', 10, 'auto')                     |
| S7     | 88.23%        | ('rbf', 10, 'auto')                     |
| S8     | 87.93%        | ('rbf', 10, 'scale')                    |
| S9     | **88.88%**    | ('rbf', 10, 'scale')                    |
| S10    | 87.62%        | ('rbf', 10, 'auto')                     |

---

## Figure 1: Convergence Graph of Best SVM (Sample S9)

![image](https://github.com/user-attachments/assets/6d40bae7-5246-48da-a30a-6d83ec78e743)


---

## Explanantion of the graph:

The convergence graph shows **accuracy vs. iteration** during random hyperparameter tuning for SVM. The reason for so many fluctuations can be explained by:

### 1. Random Hyperparameter Sampling
Each iteration tests a new random set of:
- Kernel (`'rbf'`, `'linear'`, etc.)
- `C` value
- `gamma` strategy

Some combinations perform well, others don’t — leading to varying, not fixed accuracy values.

### 2. No Learning from History
The search method is **random**, with no memory or strategy. It doesn't adapt based on past performance like grid search or Bayesian optimization.

### 3. Sensitivity to Hyperparameters
SVM's performance varies **dramatically** with small changes in `C` or `gamma`, especially with the `rbf` kernel.

### 4. Limited Data & Imbalance
Some classes may be underrepresented in certain train-test splits, further affecting model consistency.

---
