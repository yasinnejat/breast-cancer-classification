# Breast Cancer Classification Using Machine Learning

A machine-learning project that classifies breast tumor measurements as **malignant (M)** or **benign (B)** using Logistic Regression.

## Files
- `breast_cancer_classification.ipynb` — the submitted Jupyter Notebook.
- `cancer.csv` — dataset file required by the notebook.
- `requirements.txt` — Python dependencies.

## Dataset
The CSV contains 569 samples, 30 numeric features, an `id` column, and a `diagnosis` label. It is prepared from scikit-learn's built-in Wisconsin Diagnostic Breast Cancer dataset and formatted to match the notebook's expected column names and labels (`M` = malignant, `B` = benign). No separate dataset download is needed.

## Run the project
1. Install Python 3.10 or newer.
2. In this folder, install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Open `breast_cancer_classification.ipynb` in Jupyter Notebook, JupyterLab, or VS Code.
4. Run cells from top to bottom. Keep `cancer.csv` in the same folder as the notebook.

## Workflow
1. Load and inspect the data with Pandas.
2. Remove the identifier column and encode the diagnosis (`M` = 1, `B` = 0).
3. Apply min-max feature scaling.
4. Split the data into training and test sets.
5. Train a Logistic Regression classifier and evaluate predictions.

## Result
The submitted notebook reports **89.53% accuracy** on its 86-row test split (77 correct predictions out of 86). This is the result recorded by the notebook, not a guarantee of performance on new data.

## Important limitation
This is an educational project, not a medical diagnostic tool. The notebook scales the complete dataset before splitting it, which can cause data leakage. For a more reliable evaluation, fit preprocessing only on training data (for example, using a scikit-learn `Pipeline`) and consider stratified splitting or cross-validation.

## Technologies
Python, Pandas, NumPy, Matplotlib, scikit-learn, Logistic Regression, Min-Max scaling.
