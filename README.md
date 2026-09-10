# Implement Fundamental ML Algorithms from First Principles (Scratch)

A from-scratch implementation of **Linear Regression**, **Logistic Regression**, and **K-Nearest Neighbors (KNN)** using only NumPy for the core mathematics, benchmarked against their Scikit-Learn reference implementations.

## Objective

Demonstrate genuine mathematical and practical understanding of three foundational ML algorithms by implementing their training and prediction logic entirely by hand — no calls to Scikit-Learn's model classes inside the core implementations — and validating correctness against well-tested library equivalents.

## What "from scratch" means here

- Every gradient, cost function, and update rule is derived in the notebook and implemented directly with NumPy array operations.
- Scikit-Learn is used **only** for: loading benchmark datasets, `train_test_split`, `StandardScaler`, the *reference* models used for comparison, and standard evaluation metrics (accuracy, MSE, etc.). None of these substitute for the algorithms themselves.

## Algorithms implemented

| Algorithm | Training method | Dataset | Task |
|---|---|---|---|
| Linear Regression | Batch Gradient Descent | Diabetes (sklearn built-in) | Regression |
| Logistic Regression | Gradient Descent on binary cross-entropy | Breast Cancer Wisconsin (sklearn built-in) | Binary classification |
| K-Nearest Neighbors | None (instance-based) | Breast Cancer Wisconsin (sklearn built-in) | Binary classification |

## Repository structure

```
ml-from-scratch/
├── README.md
├── requirements.txt
├── notebooks/
│   └── ml_from_scratch.ipynb
├── src/                        # optional: extracted classes for reuse
│   ├── linear_regression_scratch.py
│   ├── logistic_regression_scratch.py
│   └── knn_scratch.py
└── .gitignore
```

## Running the notebook

1. Clone the repository and create a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Launch Jupyter and open the notebook:
   ```bash
   jupyter notebook notebooks/ml_from_scratch.ipynb
   ```
4. Run all cells top to bottom (**Kernel → Restart & Run All**). The notebook is fully self-contained, uses only Scikit-Learn's bundled datasets (no internet access required), and sets `random_state=42` throughout for reproducible results.

## Results summary

| Algorithm | Implementation | Main Metric(s) |
|---|---|---|
| Linear Regression | Scratch | MSE ≈ 2899, R² ≈ 0.453 |
| Linear Regression | Scikit-Learn | MSE ≈ 2900, R² ≈ 0.453 |
| Logistic Regression | Scratch | Accuracy ≈ 0.974, F1 ≈ 0.979 |
| Logistic Regression | Scikit-Learn | Accuracy ≈ 0.982, F1 ≈ 0.986 |
| KNN (k=5) | Scratch | Accuracy ≈ 0.956, F1 ≈ 0.966 |
| KNN (k=5) | Scikit-Learn | Accuracy ≈ 0.956, F1 ≈ 0.966 (100% prediction agreement) |

Small gaps between scratch and Scikit-Learn results are explained in the notebook (solver type, regularization defaults, convergence) rather than forced to match artificially.

## License

MIT (or your preferred license — update before publishing).

