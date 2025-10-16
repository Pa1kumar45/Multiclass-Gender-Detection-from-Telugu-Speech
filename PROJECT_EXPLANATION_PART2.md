# PART 2: Machine Learning Model Training & Evaluation

## PASProject2.ipynb - Detailed Explanation

## 📋 Overview

This notebook is the **second phase** of the Gender Detection project. While PASProject1 extracted features from audio files, this notebook:

1. **Loads** the extracted features from Excel
2. **Trains** a Random Forest classifier
3. **Evaluates** model performance
4. **Visualizes** feature importance
5. **Makes predictions** on the entire dataset
6. **Saves** comparison results
---

## 🔄 Complete Workflow

```
PASProject1.ipynb → PASProject2.ipynb
(Feature Extraction)  (Model Training)
        ↓                    ↓
PAS_Features(new).xlsx → ML Model → Predictions
   (113 columns)        (RF Classifier)  (comparison_results)
```

---

## 📝 Code Walkthrough - Cell by Cell

###

**Cell 1: Import Machine Learning Libraries**

```python
import pandas as pd
```

**Already familiar from Part 1:**

- Data manipulation library
- Used here to: load Excel, create DataFrames, organize results

```python
from sklearn.model_selection import train_test_split
```

**Scikit-learn module for data splitting:**

- **Purpose**: Split data into training and testing sets
- **Why needed**: Evaluate model on unseen data (avoid overfitting)
- **Function**: `train_test_split()` - randomly divides dataset
- **Best practice**: Hold out 20-30% for testing

```python
from sklearn.ensemble import RandomForestClassifier
```

**Random Forest algorithm import:**

- **Type**: Ensemble learning method (combines multiple decision trees)
- **Advantages**:
  - High accuracy for classification
  - Handles non-linear relationships
  - Robust to overfitting (compared to single decision tree)
  - Provides feature importance scores
- **How it works**:
  - Trains N decision trees on random subsets of data
  - Each tree votes on the prediction
  - Final prediction = majority vote

```python
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
```

**Evaluation metrics import:**

1. **`accuracy_score`**:

   - Formula: `(Correct Predictions) / (Total Predictions)`
   - Range: 0.0 to 1.0 (0% to 100%)
   - Simple overall performance measure

2. **`classification_report`**:
   - Detailed metrics per class
   - Includes: Precision, Recall, F1-Score, Support
   - Helps identify class-specific performance

3. **`confusion_matrix`**:
   - 2×2 matrix for binary classification
   - Shows: True Positives, False Positives, True Negatives, False Negatives
   - Visual way to see where model makes mistakes

```python
import matplotlib.pyplot as plt
```

**Matplotlib plotting library:**

- **Purpose**: Create visualizations (bar charts, plots)
- **Alias**: `plt` is standard convention
- **Usage in notebook**: Plot feature importance bar chart
- **Key method**: `plt.figure()`, `plt.barh()`, `plt.show()`

```python
from google.colab import drive
```

**Google Colab import (again):**

- Same as PASProject1
- Needed to access Google Drive where Excel file is stored
- **⚠️ Note**: Remove for local execution

---

### **Cell 2: Mount Google Drive**

```python
drive.mount("/content/drive", force_remount=True)
```

**Identical to PASProject1 Cell 2:**

- Mounts Google Drive to `/content/drive`
- `force_remount=True` ensures fresh connection
- Allows access to `PAS_Features(new).xlsx` file

**For local execution**, replace with:

```python
# No mounting needed for local files
# Excel file should be in same directory as notebook
```

---

### **Cell 3: Load Feature Data**

```python
file_path = '/content/drive/MyDrive/PAS_Features(new).xlsx'
```

**Define Excel file path:**

- **Location**: Google Drive root → `PAS_Features(new).xlsx`
- **This file**: Created by PASProject1.ipynb
- **Contents**: 113 columns × N rows (audio features + labels)

**For local execution**, use:

```python
file_path = 'PAS_Features(new).xlsx'  # Same directory
# OR
file_path = r'C:\full\path\to\PAS_Features(new).xlsx'
```

```python
feature_df = pd.read_excel(file_path)
```

**Load Excel into DataFrame:**

- **`pd.read_excel()`**: Pandas function to read Excel files
  - Requires `openpyxl` library (already installed)
  - Automatically detects sheet, headers, data types
- **`feature_df`**: DataFrame variable holding all features
- **Shape**: `(N_samples, 113)` where N = total audio files
- **Columns**: `['audio_file', 'pitch', 'mfcc1', ..., 'mfcc110', 'gender']`

**What happens internally:**

1. Opens .xlsx file
2. Reads first sheet
3. First row → column names
4. Remaining rows → data
5. Returns pandas DataFrame object

---

### **Cell 4: Prepare Data for Machine Learning**

```python
X = feature_df.drop(columns=['audio_file', 'gender'])
```

**Create feature matrix (X):**

- **`feature_df.drop()`**: Remove specified columns
  - **`columns=['audio_file', 'gender']`**: Drop these 2 columns
    - `audio_file`: Just filepath (not a predictive feature)
    - `gender`: This is the **label** we want to predict (target variable)
- **Result**: `X` contains only the **111 numeric features**
  - Column 0: `pitch` (1 feature)
  - Columns 1-110: `mfcc1` to `mfcc110` (110 features)
- **Shape**: `(N_samples, 111)`
- **Purpose**: Input features for ML model

**Machine Learning Terminology:**

- **X**: Feature matrix (independent variables, predictors)
- **y**: Target vector (dependent variable, labels)

```python
y = feature_df['gender']
```

**Create target vector (y):**

- **`feature_df['gender']`**: Extract only the gender column
- **Contents**: Array of strings: 'male' or 'female'
- **Shape**: `(N_samples,)` - 1D array
- **Purpose**: What we want the model to predict

**Example:**

```python
# X contains:
#   pitch  mfcc1  mfcc2  ...  mfcc110
# [ 218.5, -142.3, 25.1, ..., 0.3 ]  ← Female sample
# [ 125.2, -138.1, 22.4, ..., 0.1 ]  ← Male sample

# y contains:
# ['female', 'male', 'female', ...]
```

```python
# Train-test split
```

**Comment indicating next step:**

- Standard ML practice: split data before training
- Prevents information leakage from test set to training

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

**Split data into training and testing sets:**

**Function signature breakdown:**

- **`train_test_split(X, y, ...)`**: Main function
  - Returns 4 arrays: X_train, X_test, y_train, y_test

**Parameters explained:**

1. **`X, y`**: Input feature matrix and target vector

   - Data to be split

2. **`test_size=0.2`**:

   - **Meaning**: 20% of data for testing, 80% for training
   - **Example**: 500 samples → 400 train, 100 test
   - **Why 20%?** Standard practice (provides good balance)
   - **Alternatives**: 0.25 (25%), 0.3 (30%)

3. **`random_state=42`**:

   - **Purpose**: Seed for random number generator
   - **Effect**: Makes split reproducible (same split every run)
   - **Why 42?** Arbitrary convention (reference to "Hitchhiker's Guide")
   - **Without it**: Different split each run (harder to debug)

4. **`stratify=y`**:
   - **Purpose**: Maintain class distribution in both sets
   - **Example**: If dataset is 60% female, 40% male:
     - Training: 60% female, 40% male
     - Testing: 60% female, 40% male
   - **Why important?** Prevents class imbalance in test set
   - **Without it**: Random split might give 70% female in train, 50% in test

**Output variables:**

| Variable  | Contents          | Shape        | Purpose        |
| --------- | ----------------- | ------------ | -------------- |
| `X_train` | Training features | (0.8×N, 111) | Train model    |
| `X_test`  | Testing features  | (0.2×N, 111) | Evaluate model |
| `y_train` | Training labels   | (0.8×N,)     | Train model    |
| `y_test`  | Testing labels    | (0.2×N,)     | Evaluate model |

**Example with 500 samples:**

- X_train: (400, 111) - 400 samples for training
- X_test: (100, 111) - 100 samples for testing
- y_train: (400,) - 400 labels for training
- y_test: (100,) - 100 labels for testing

**Why split data?**

- **Training set**: Model learns patterns from this data
- **Testing set**: Model never sees this during training
- **Evaluation**: Test accuracy shows real-world performance
- **Prevents overfitting**: Model can't memorize test data

---

### **Cell 5: Train Random Forest Model**

```python
classifier = RandomForestClassifier(
    n_estimators=200, max_depth=20, random_state=42, class_weight='balanced'
)
```

**Initialize Random Forest classifier:**

**Hyperparameters explained:**

1. **`n_estimators=200`**:

   - **Meaning**: Build 200 decision trees
   - **Each tree**: Trained on random subset of data (bootstrap sampling)
   - **More trees**:
     - ✅ More accurate (to a point)
     - ❌ Slower training
     - ❌ More memory
   - **Typical values**: 100-500
   - **Default**: 100

2. **`max_depth=20`**:

   - **Meaning**: Each tree can be at most 20 levels deep
   - **Deeper trees**:
     - ✅ Can model complex patterns
     - ❌ Risk overfitting
   - **Shallower trees**:
     - ✅ Faster, prevent overfitting
     - ❌ May underfit
   - **Value 20**: Good balance for this dataset
   - **Default**: None (unlimited depth)

3. **`random_state=42`**:

   - **Purpose**: Reproducible results
   - **Effect**: Same trees built every run
   - **Why needed**: Debugging, comparing experiments

4. **`class_weight='balanced'`**:
   - **Purpose**: Handle class imbalance automatically
   - **How it works**:
     ```
     Weight = N_samples / (N_classes × N_samples_per_class)
     ```
   - **Example**: 300 female, 200 male samples
     - Female weight: 500 / (2 × 300) = 0.833
     - Male weight: 500 / (2 × 200) = 1.25
   - **Effect**: Minority class errors penalized more
   - **Alternative**: `class_weight={0: 1, 1: 2}` (manual weights)

**Random Forest Algorithm Overview:**

```
Forest = Tree1 + Tree2 + ... + Tree200

Each Tree:
1. Randomly sample data (bootstrap)
2. At each split, randomly select subset of features
3. Choose best feature/threshold from subset
4. Repeat until max_depth or pure leaf

Prediction:
- Each tree votes: 'male' or 'female'
- Final prediction = majority vote
- Example: 130 trees vote 'female', 70 vote 'male' → 'female'
```

```python
classifier.fit(X_train, y_train)
```

**Train the model:**

- **`.fit()`**: Scikit-learn method to train classifier
  - **`X_train`**: Training features (400 samples × 111 features)
  - **`y_train`**: Training labels (400 labels)
- **What happens**:
  1. For each of 200 trees:
     - Randomly sample ~63% of training data (bootstrap)
     - Build decision tree using random feature subsets
     - Store trained tree in `classifier` object
  2. Return trained classifier
- **Duration**: ~1-5 seconds depending on dataset size
- **Output**: Trained `classifier` object (ready to predict)

**Internal process for one tree:**

```
Root node: All 400 training samples
├─ Split on mfcc3 < 15.2?
│  ├─ Yes: 250 samples (mostly female)
│  │  └─ Split on pitch < 180?
│  │     ├─ Yes: 200 → LEAF: Female
│  │     └─ No: 50 → LEAF: Male
│  └─ No: 150 samples (mostly male)
│     └─ Split on mfcc7 < -5.1?
│        ├─ Yes: 120 → LEAF: Male
│        └─ No: 30 → LEAF: Female
... (continues to max_depth=20)
```

```python
# Predict and evaluate
```

**Comment for next steps:**

- Use trained model to make predictions
- Evaluate accuracy on test set

```python
y_pred = classifier.predict(X_test)
```

**Make predictions on test set:**

- **`.predict()`**: Use trained model to predict labels
  - **`X_test`**: Testing features (100 samples × 111 features)
  - Model has NEVER seen these samples during training
- **Process**:
  1. For each test sample:
     - Pass through all 200 trees
     - Each tree outputs: 'male' or 'female'
     - Count votes (e.g., 145 'female', 55 'male')
     - Final prediction = majority (e.g., 'female')
  2. Return array of predictions
- **`y_pred`**: Predicted labels (100 predictions)
  - Shape: `(100,)` matching `y_test`
  - Example: `['female', 'male', 'female', 'female', ...]`

```python
accuracy = accuracy_score(y_test, y_pred)
```

**Calculate accuracy:**

- **`accuracy_score()`**: Compare predictions to true labels
  - **`y_test`**: True labels (100 ground truth labels)
  - **`y_pred`**: Predicted labels (100 model predictions)
- **Formula**:
  ```
  Accuracy = (Number of correct predictions) / (Total predictions)
           = (TP + TN) / (TP + TN + FP + FN)
  ```
- **`accuracy`**: Float between 0.0 and 1.0
  - Example: 0.94 = 94% accuracy (94 out of 100 correct)
- **Stored for display in next cell**

**Example calculation:**

```python
y_test = ['female', 'male', 'female', 'male', ...]  # 100 labels
y_pred = ['female', 'male', 'female', 'female', ...]  # 100 predictions
         #   ✓       ✓        ✓         ✗

# 94 correct, 6 incorrect
# accuracy = 94/100 = 0.94
```

---

### **Cell 6: Display Evaluation Metrics**

```python
print(f"\nModel Accuracy: {accuracy*100:.2f}%")
```

**Print accuracy percentage:**

- **`\n`**: Newline for spacing
- **f-string**: Format string with variable
- **`accuracy*100`**: Convert decimal to percentage (0.94 → 94)
- **`:.2f`**: Format to 2 decimal places (94.00%)
- **Example output**: `Model Accuracy: 94.00%`

```python
print("\nClassification Report:\n", classification_report(y_test, y_pred))
```

**Print detailed classification metrics:**

**`classification_report()` output format:**

```
              precision    recall  f1-score   support

      female       0.96      0.92      0.94        50
        male       0.92      0.96      0.94        50

    accuracy                           0.94       100
   macro avg       0.94      0.94      0.94       100
weighted avg       0.94      0.94      0.94       100
```

**Metrics explained:**

1. **Precision** (per class):

   - Formula: `TP / (TP + FP)`
   - **Female precision 0.96**: Of all predicted females, 96% were actually female
   - **Interpretation**: How many selected items are relevant?
   - **High precision**: Few false positives

2. **Recall** (Sensitivity, True Positive Rate):

   - Formula: `TP / (TP + FN)`
   - **Female recall 0.92**: Of all actual females, 92% were correctly identified
   - **Interpretation**: How many relevant items are selected?
   - **High recall**: Few false negatives

3. **F1-Score**:

   - Formula: `2 × (Precision × Recall) / (Precision + Recall)`
   - **Harmonic mean** of precision and recall
   - **Range**: 0 to 1 (1 is perfect)
   - **Why useful**: Single metric balancing precision and recall

4. **Support**:

   - **Number of actual samples** in each class
   - Female support 50: 50 female samples in test set
   - Male support 50: 50 male samples in test set
   - Helps interpret metrics (metrics on 5 samples less reliable than 500)

5. **Macro avg**:

   - **Unweighted mean** of per-class metrics
   - Treats all classes equally (even if imbalanced)

6. **Weighted avg**:
   - **Weighted by support** (number of samples per class)
   - Better for imbalanced datasets

**Real example:**

```
True labels (y_test): 50 female, 50 male
Predictions (y_pred):
- Predicted 52 as female: 48 correct (TP), 4 incorrect (FP)
- Predicted 48 as male: 46 correct (TN), 2 incorrect (FN)

Female:
  Precision = 48/(48+4) = 0.923
  Recall = 48/(48+2) = 0.960
  F1 = 2×(0.923×0.960)/(0.923+0.960) = 0.941
```

```python
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
```

**Print confusion matrix:**

**Confusion Matrix structure (2×2 for binary classification):**

```
                 Predicted
                 Female  Male
Actual Female  [  48      2  ]  ← 48 correct, 2 misclassified
       Male    [   4     46  ]  ← 4 misclassified, 46 correct
```

**Matrix interpretation:**

| Position | Name                | Meaning                    | Example Value |
| -------- | ------------------- | -------------------------- | ------------- |
| [0,0]    | True Positive (TP)  | Correctly predicted female | 48            |
| [0,1]    | False Negative (FN) | Female predicted as male   | 2             |
| [1,0]    | False Positive (FP) | Male predicted as female   | 4             |
| [1,1]    | True Negative (TN)  | Correctly predicted male   | 46            |

**Calculations from confusion matrix:**

```
Accuracy = (TP + TN) / Total = (48 + 46) / 100 = 0.94
Precision (Female) = TP / (TP + FP) = 48 / (48 + 4) = 0.923
Recall (Female) = TP / (TP + FN) = 48 / (48 + 2) = 0.960
Specificity = TN / (TN + FP) = 46 / (46 + 4) = 0.920
```

**What to look for:**

- **Diagonal values (TP, TN)**: Should be high (correct predictions)
- **Off-diagonal (FP, FN)**: Should be low (errors)
- **Class imbalance**: If one row much larger, may need rebalancing
- **Error patterns**: Is model biased toward one class?

---

### **Cell 7: Feature Importance Visualization**

```python
importances = classifier.feature_importances_
```

**Extract feature importance scores:**

- **`.feature_importances_`**: Attribute of trained Random Forest
- **What it measures**: How useful each feature was for prediction
- **Calculation**:
  - For each tree, track which features were used for splits
  - Features used for early/important splits get higher scores
  - Average across all 200 trees
- **Output**: Array of 111 values (one per feature)
  - Shape: `(111,)` matching `X.columns`
  - Example: `[0.245, 0.003, 0.018, 0.142, ...]`
  - Sum of all importances = 1.0

**Importance interpretation:**

- **High value (e.g., 0.25)**: Very important for predictions (pitch often high)
- **Low value (e.g., 0.001)**: Barely used (some MFCCs may be redundant)
- **Zero**: Never used for splitting

```python
feat_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)
```

**Create and sort feature importance DataFrame:**

**Step-by-step breakdown:**

1. **`pd.DataFrame({ ... })`**: Create DataFrame with 2 columns
2. **`'Feature': X.columns`**:

   - Column 1: Feature names
   - `X.columns` contains: `['pitch', 'mfcc1', 'mfcc2', ..., 'mfcc110']`
   - 111 feature names

3. **`'Importance': importances`**:

   - Column 2: Importance scores
   - Values from `classifier.feature_importances_`
   - 111 importance values

4. **`.sort_values(by='Importance', ascending=False)`**:
   - **`by='Importance'`**: Sort by Importance column
   - **`ascending=False`**: Highest importance first
   - **Result**: Top features at the top

**Resulting DataFrame:**

```
     Feature  Importance
0      pitch      0.2450
23     mfcc23     0.1420
15     mfcc15     0.0980
...      ...         ...
108  mfcc108     0.0001
```

```python
# top 10 important features
```

**Comment**: Next lines visualize top 10 features

```python
plt.figure(figsize=(10,6))
```

**Create figure for plot:**

- **`plt.figure()`**: Initialize new plot
- **`figsize=(10,6)`**: Set size in inches (width=10, height=6)
  - Creates appropriately sized plot for readability

```python
plt.barh(feat_importance_df['Feature'][:10], feat_importance_df['Importance'][:10], color='skyblue')
```

**Create horizontal bar chart:**

- **`plt.barh()`**: Horizontal bar plot function
  - **First arg**: `feat_importance_df['Feature'][:10]`
    - Y-axis labels (feature names)
    - `[:10]` slices first 10 rows (top 10 after sorting)
  - **Second arg**: `feat_importance_df['Importance'][:10]`
    - Bar lengths (importance values)
    - `[:10]` first 10 importance scores
  - **`color='skyblue'`**: Bar color (aesthetic)
- **Result**: 10 horizontal bars, length = importance

**Example visualization data:**

```
pitch     ■■■■■■■■■■■■■■■■■■■■■■■■■ 0.245
mfcc23    ■■■■■■■■■■■■■■■ 0.142
mfcc15    ■■■■■■■■■■ 0.098
mfcc7     ■■■■■■■■ 0.075
mfcc45    ■■■■■■ 0.061
...
```

```python
plt.gca().invert_yaxis()
```

**Invert Y-axis:**

- **`plt.gca()`**: Get Current Axes (the plot axes)
- **`.invert_yaxis()`**: Flip Y-axis direction
- **Effect**: Top feature appears at top (instead of bottom)
- **Without this**: Most important feature at bottom of chart

```python
plt.xlabel("Importance")
```

**Set X-axis label:**

- **`plt.xlabel()`**: Add label to X-axis
- **Text**: "Importance"
- **Purpose**: Clarify what horizontal length represents

```python
plt.title("Top 10 Important Features")
```

**Set plot title:**

- **`plt.title()`**: Add title above plot
- **Text**: "Top 10 Important Features"
- **Purpose**: Describe plot content

```python
plt.show()
```

**Display the plot:**

- **`plt.show()`**: Render and show the plot
- **In notebook**: Displays inline below cell
- **Effect**: Shows completed bar chart

**Complete plot appearance:**

```
        Top 10 Important Features

pitch     ■■■■■■■■■■■■■■■■■■■■■■■■■
mfcc23    ■■■■■■■■■■■■■■■
mfcc15    ■■■■■■■■■■
mfcc7     ■■■■■■■■
mfcc45    ■■■■■■
mfcc89    ■■■■■
mfcc3     ■■■■
mfcc52    ■■■■
mfcc11    ■■■
mfcc67    ■■■
          0.00    0.10    0.20    0.30
                  Importance
```

---

### **Cell 8: Make Predictions on Full Dataset**

```python
feature_df['Predicted_Gender'] = classifier.predict(X)
```

**Predict gender for ALL samples (including training data):**

**Breakdown:**

1. **`classifier.predict(X)`**:

   - **`X`**: ENTIRE feature matrix (all samples, not just test)
   - Includes both training and testing data
   - Shape: `(N_total, 111)` where N_total = all audio files
   - **Returns**: Predictions for every sample

2. **`feature_df['Predicted_Gender'] = ...`**:
   - Create new column in original DataFrame
   - Column name: `'Predicted_Gender'`
   - Contains model predictions for all samples
   - Now `feature_df` has 114 columns (original 113 + new column)

**Why predict on full dataset?**

- Compare actual vs predicted for every audio file
- Identify which specific files were misclassified
- Useful for error analysis and debugging

**Resulting DataFrame structure:**

```
audio_file  pitch  mfcc1 ... mfcc110  gender  Predicted_Gender
file001.wav 218.5  -142.3... 0.3      female  female
file002.wav 125.2  -138.1... 0.1      male    male
file003.wav 205.1  -140.2... 0.2      female  female
file004.wav 132.8  -139.5... 0.15     male    female  ← Error!
```

```python
comparison_df = pd.DataFrame({
    'Actual': y,
    'Predicted': feature_df['Predicted_Gender']
})
```

**Create comparison DataFrame:**

**Structure:**

1. **`pd.DataFrame({ ... })`**: Create new DataFrame
2. **`'Actual': y`**:

   - Column 1: True labels
   - `y` = `feature_df['gender']` (original labels)
   - Example: `['female', 'male', 'female', ...]`

3. **`'Predicted': feature_df['Predicted_Gender']`**:
   - Column 2: Model predictions
   - Values we just computed above
   - Example: `['female', 'male', 'female', ...]`

**Result**: Simple 2-column DataFrame for comparison

```
     Actual  Predicted
0    female     female  ✓
1      male       male  ✓
2    female     female  ✓
3      male     female  ✗ (Error)
4      male       male  ✓
...
```

```python
print("\nSample Comparison:")
print(comparison_df.head(10))
```

**Display first 10 comparisons:**

- **`comparison_df.head(10)`**: Get first 10 rows
- **`print()`**: Display to console
- **Purpose**: Quick visual check of predictions vs actual

**Sample output:**

```
Sample Comparison:
     Actual  Predicted
0    female     female
1      male       male
2    female     female
3      male       male
4    female     female
5      male     female  ← Misclassification
6    female     female
7      male       male
8    female     female
9      male       male
```

---

### **Cell 9: Save Comparison Results**

```python
comparison_df.to_csv('/content/drive/MyDrive/comparison_results_PAS(new).csv', index=False)
```

**Export comparison to CSV:**

**Breakdown:**

1. **`.to_csv()`**: Pandas method to write CSV file

   - **Format**: Comma-Separated Values (simpler than Excel)
   - **Advantage**: Smaller file size, faster read/write

2. **`'/content/drive/MyDrive/comparison_results_PAS(new).csv'`**:

   - **Path**: Google Drive root
   - **Filename**: `comparison_results_PAS(new).csv`
   - **Extension**: `.csv` (text format, can open in Excel)

3. **`index=False`**:
   - **Don't include row numbers** as a column
   - Without this: CSV would have extra column (0, 1, 2, ...)
   - With this: Only 'Actual' and 'Predicted' columns

**CSV file contents:**

```csv
Actual,Predicted
female,female
male,male
female,female
male,female
...
```

**For local execution:**

```python
comparison_df.to_csv('comparison_results_PAS(new).csv', index=False)
```

```python
print("\nComparison results saved as 'comparison_results_PAS(new).csv'")
```

**Confirmation message:**

- Notify user of successful save
- Shows filename

---

## 🎯 Machine Learning Concepts Deep Dive

### 1. **Random Forest Algorithm**

#### How It Works

**Single Decision Tree:**

```
           [pitch < 180?]
              /     \
           Yes       No
           /           \
    [mfcc3 < 15?]   [mfcc7 < -5?]
       /    \          /      \
   Female  Male    Male    Female
```

**Random Forest = Ensemble of Trees:**

```
Tree 1:        Tree 2:        Tree 3:        ...  Tree 200:
Female         Male           Female              Female
   ↓              ↓              ↓                   ↓
                Vote Aggregation
                       ↓
           Female (145 votes) vs Male (55 votes)
                       ↓
                Final: Female
```

#### Advantages

1. **Reduced Overfitting**:

   - Single tree: Can memorize training data
   - Forest: Voting averages out individual tree errors

2. **Feature Importance**:

   - Tracks which features are used most often
   - Helps understand what drives predictions

3. **Handles Non-linearity**:

   - Can model complex decision boundaries
   - No assumption about feature relationships

4. **Robust to Outliers**:
   - Outliers affect only some trees
   - Voting reduces their impact

#### Hyperparameter Tuning

| Parameter      | Value Used | Effect                                           | Alternatives            |
| -------------- | ---------- | ------------------------------------------------ | ----------------------- |
| `n_estimators` | 200        | More trees = more accurate (diminishing returns) | 100, 500, 1000          |
| `max_depth`    | 20         | Limits tree complexity                           | None (no limit), 10, 30 |
| `class_weight` | 'balanced' | Equal importance to both classes                 | None, {0:1, 1:2}        |
| `random_state` | 42         | Reproducibility                                  | Any integer             |

---

### 2. **Evaluation Metrics Comparison**

| Metric          | Formula                 | When to Use           | Example |
| --------------- | ----------------------- | --------------------- | ------- |
| **Accuracy**    | `(TP+TN)/(TP+TN+FP+FN)` | Balanced classes      | 94%     |
| **Precision**   | `TP/(TP+FP)`            | Minimize false alarms | 92.3%   |
| **Recall**      | `TP/(TP+FN)`            | Don't miss positives  | 96.0%   |
| **F1-Score**    | `2×P×R/(P+R)`           | Balance P and R       | 94.1%   |
| **Specificity** | `TN/(TN+FP)`            | Correct negatives     | 92.0%   |

---

## 📊 Complete Pipeline Summary

### Project Files Overview

| File                  | Purpose                  | Input                  | Output                          |
| --------------------- | ------------------------ | ---------------------- | ------------------------------- |
| **PASProject1.ipynb** | Feature extraction       | Audio WAV files        | PAS_Features(new).xlsx          |
| **PASProject2.ipynb** | ML training & evaluation | PAS_Features(new).xlsx | comparison_results_PAS(new).csv |

### End-to-End Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│ Raw Data: Audio Files                                       │
│ - Male voices: 200-400 WAV files                           │
│ - Female voices: 200-400 WAV files                         │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ↓ PASProject1.ipynb
                      │
┌─────────────────────▼───────────────────────────────────────┐
│ Extracted Features: PAS_Features(new).xlsx                  │
│ - Rows: N audio files (e.g., 500)                          │
│ - Columns: 113 (filepath, pitch, 110 MFCCs, gender)        │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ↓ PASProject2.ipynb
                      │
┌─────────────────────▼───────────────────────────────────────┐
│ Machine Learning Pipeline                                    │
│                                                              │
│ 1. Load Data (pd.read_excel)                               │
│ 2. Split: 80% train, 20% test                              │
│ 3. Train Random Forest (200 trees)                         │
│ 4. Evaluate: 94% accuracy                                  │
│ 5. Feature Importance: pitch most important                │
│ 6. Full predictions on all samples                         │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ↓
┌─────────────────────▼───────────────────────────────────────┐
│ Outputs                                                      │
│ - Console: Metrics (accuracy, precision, recall, F1)       │
│ - Plot: Top 10 feature importance bar chart                │
│ - CSV: comparison_results_PAS(new).csv                     │
│         (Actual vs Predicted for all samples)               │
└─────────────────────────────────────────────────────────────┘
```

---

## 🚀 Advanced Topics & Extensions

### 1. **Hyperparameter Optimization**

```python
from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'n_estimators': [100, 200, 500],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Grid search with cross-validation
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,  # 5-fold cross-validation
    scoring='accuracy',
    n_jobs=-1  # Use all CPU cores
)

grid_search.fit(X_train, y_train)

# Best parameters
print("Best params:", grid_search.best_params_)
print("Best CV score:", grid_search.best_score_)

# Use best model
best_classifier = grid_search.best_estimator_
```

---

### 2. **Alternative ML Algorithms**

#### Support Vector Machine (SVM)

```python
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# SVM requires feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train SVM
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale')
svm_model.fit(X_train_scaled, y_train)

# Evaluate
y_pred_svm = svm_model.predict(X_test_scaled)
print("SVM Accuracy:", accuracy_score(y_test, y_pred_svm))
```

#### Neural Network

```python
from sklearn.neural_network import MLPClassifier

# Train neural network
nn_model = MLPClassifier(
    hidden_layer_sizes=(100, 50),  # 2 hidden layers
    activation='relu',
    max_iter=500,
    random_state=42
)
nn_model.fit(X_train, y_train)

# Evaluate
y_pred_nn = nn_model.predict(X_test)
print("NN Accuracy:", accuracy_score(y_test, y_pred_nn))
```

#### Gradient Boosting

```python
from sklearn.ensemble import GradientBoostingClassifier

# Train gradient boosting
gb_model = GradientBoostingClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.1,
    random_state=42
)
gb_model.fit(X_train, y_train)

# Evaluate
y_pred_gb = gb_model.predict(X_test)
print("GB Accuracy:", accuracy_score(y_test, y_pred_gb))
```

---

### 3. **Advanced Visualizations**

#### ROC Curve

```python
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# Get probability predictions (need binary encoding first)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_test_encoded = le.fit_transform(y_test)  # 0=female, 1=male

y_proba = classifier.predict_proba(X_test)[:, 1]  # Probability of class 1

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test_encoded, y_proba)
roc_auc = auc(fpr, tpr)

# Plot
plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, color='darkorange', lw=2,
         label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.show()
```

#### Learning Curve

```python
from sklearn.model_selection import learning_curve

# Calculate learning curve
train_sizes, train_scores, test_scores = learning_curve(
    RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42),
    X, y,
    cv=5,
    train_sizes=np.linspace(0.1, 1.0, 10),
    n_jobs=-1
)

# Plot
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.figure(figsize=(10,6))
plt.plot(train_sizes, train_mean, label='Training score', color='blue')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                 alpha=0.1, color='blue')
plt.plot(train_sizes, test_mean, label='Cross-validation score', color='green')
plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std,
                 alpha=0.1, color='green')
plt.xlabel('Training Set Size')
plt.ylabel('Accuracy Score')
plt.title('Learning Curve')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.show()
```

#### T-SNE Visualization

```python
from sklearn.manifold import TSNE

# Reduce to 2D
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_2d = tsne.fit_transform(X)

# Plot
plt.figure(figsize=(10,8))
colors = {'male': 'blue', 'female': 'red'}
for gender in ['male', 'female']:
    mask = y == gender
    plt.scatter(X_2d[mask, 0], X_2d[mask, 1],
                c=colors[gender], label=gender, alpha=0.6, s=50)
plt.xlabel('T-SNE Component 1')
plt.ylabel('T-SNE Component 2')
plt.title('T-SNE Visualization of Audio Features')
plt.legend()
plt.grid(alpha=0.3)
plt.show()
```

---

### 4. **Error Analysis**

```python
# Find misclassified samples
errors_mask = comparison_df['Actual'] != comparison_df['Predicted']
errors = comparison_df[errors_mask]
error_indices = errors.index

print(f"\nTotal errors: {len(errors)} out of {len(comparison_df)}")
print(f"Error rate: {len(errors)/len(comparison_df)*100:.2f}%\n")

# Analyze pitch of errors
error_pitch = feature_df.loc[error_indices, 'pitch']
correct_pitch = feature_df.loc[~errors_mask, 'pitch']

print("Pitch statistics:")
print(f"Error samples - Mean: {error_pitch.mean():.1f} Hz, Std: {error_pitch.std():.1f} Hz")
print(f"Correct samples - Mean: {correct_pitch.mean():.1f} Hz, Std: {correct_pitch.std():.1f} Hz")

# Show misclassified files
print("\nMisclassified files:")
for idx in error_indices[:10]:  # Show first 10
    row = feature_df.loc[idx]
    print(f"{row['audio_file']}: Actual={row['gender']}, "
          f"Predicted={row['Predicted_Gender']}, Pitch={row['pitch']:.1f} Hz")
```

---

### 5. **Cross-Validation**

```python
from sklearn.model_selection import cross_val_score, StratifiedKFold

# 5-fold stratified cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Evaluate with cross-validation
cv_scores = cross_val_score(
    RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42),
    X, y,
    cv=skf,
    scoring='accuracy',
    n_jobs=-1
)

print("Cross-validation scores:")
for fold, score in enumerate(cv_scores, 1):
    print(f"  Fold {fold}: {score*100:.2f}%")
print(f"\nMean accuracy: {cv_scores.mean()*100:.2f}%")
print(f"Std deviation: {cv_scores.std()*100:.2f}%")
```

---

### 6. **Model Deployment**

#### Save Model

```python
import joblib

# Save trained model
joblib.dump(classifier, 'gender_classifier_rf.pkl')
print("Model saved as 'gender_classifier_rf.pkl'")

# Save scaler if used
# joblib.dump(scaler, 'scaler.pkl')
```

#### Load and Use Model

```python
# Load model
loaded_classifier = joblib.load('gender_classifier_rf.pkl')

# Make prediction on new data
new_features = [[220.5, -145.2, 23.1, ...]]  # 111 features
prediction = loaded_classifier.predict(new_features)
probability = loaded_classifier.predict_proba(new_features)

print(f"Predicted gender: {prediction[0]}")
print(f"Confidence: {probability[0].max()*100:.1f}%")
```

#### Real-Time Prediction Function

```python
def predict_gender_from_audio(audio_file_path, model):
    """
    Predict gender from audio file.

    Args:
        audio_file_path: Path to WAV file
        model: Trained classifier

    Returns:
        Dictionary with prediction and confidence
    """
    # Load audio
    y, sr = librosa.load(audio_file_path, sr=16000, mono=True)

    # Extract features
    pitch = get_pitch(y, sr)
    mfccs = get_mfcc(y, sr)

    # Combine features
    features = np.array([[pitch] + mfccs.tolist()])

    # Predict
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    confidence = probabilities.max()

    return {
        'gender': prediction,
        'confidence': confidence,
        'pitch': pitch,
        'probabilities': {
            'female': probabilities[0] if model.classes_[0] == 'female' else probabilities[1],
            'male': probabilities[1] if model.classes_[1] == 'male' else probabilities[0]
        }
    }

# Example usage
result = predict_gender_from_audio('new_voice.wav', loaded_classifier)
print(f"Gender: {result['gender']}")
print(f"Confidence: {result['confidence']*100:.1f}%")
print(f"Pitch: {result['pitch']:.1f} Hz")
```

---

## 📚 Summary & Key Takeaways

### What You Learned

#### Part 1 (PASProject1):

✅ Audio loading and preprocessing  
✅ Pitch extraction using YIN algorithm  
✅ MFCC feature extraction  
✅ Feature engineering for ML  
✅ Data organization (Excel export)

#### Part 2 (PASProject2):

✅ Train-test split methodology  
✅ Random Forest classifier training  
✅ Model evaluation (accuracy, precision, recall, F1)  
✅ Confusion matrix interpretation  
✅ Feature importance analysis  
✅ Prediction workflow

### Performance Expectations

| Metric        | Typical Value | Interpretation                              |
| ------------- | ------------- | ------------------------------------------- |
| Accuracy      | 90-98%        | Very high for gender (bimodal distribution) |
| Precision     | 88-96%        | Few false positives                         |
| Recall        | 88-96%        | Few false negatives                         |
| F1-Score      | 88-96%        | Well-balanced performance                   |
| Training Time | 2-10 seconds  | Fast on moderate datasets                   |

### Why Gender Detection Works Well

1. **Clear Feature Separation**:

   - Pitch: Males ~120 Hz, Females ~220 Hz
   - Nearly non-overlapping distributions

2. **Rich Feature Set**:

   - 111 features (pitch + 110 MFCCs)
   - Captures multiple aspects of voice

3. **Robust Algorithm**:

   - Random Forest handles non-linearity
   - Ensemble reduces overfitting

4. **Balanced Dataset**:
   - Equal male/female samples
   - Stratified splitting maintains balance

---

## 🎓 Further Learning

### Next Steps

1. **Experiment**:

   - Try different algorithms (SVM, Neural Networks)
   - Tune hyperparameters (Grid Search)
   - Add more features (delta MFCCs, formants)

2. **Deploy**:

   - Create web app (Flask/Streamlit)
   - Real-time prediction from microphone
   - Mobile app integration

3. **Extend**:
   - Multi-class: Add age groups, accents
   - Speaker identification
   - Emotion recognition

### Resources

- **Scikit-learn**: https://scikit-learn.org/stable/
- **Random Forest Guide**: https://scikit-learn.org/stable/modules/ensemble.html#random-forests
- **Model Evaluation**: https://scikit-learn.org/stable/modules/model_evaluation.html
- **Visualization**: https://matplotlib.org/stable/tutorials/index.html

---

**Document Created**: Comprehensive explanation of PASProject2.ipynb  
**Topics Covered**: ML pipeline, Random Forest, evaluation metrics, feature importance, predictions  
**Complement To**: PROJECT_EXPLANATION.md (Part 1 - Feature Extraction)  
**Course**: IT 302 - Probability & Statistics Lab

---

_End of Part 2 Documentation_
