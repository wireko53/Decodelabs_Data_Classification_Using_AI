# Data Classification using KNN

Project 2 of my AI Internship at DecodeLabs.

## What I Built

A K-Nearest Neighbors classifier that identifies flower species from physical measurements, using the classic Iris dataset as a clean, well-understood benchmark for practicing the full classification workflow.

## Tech Stack

- Python
- scikit-learn (KNN, preprocessing, evaluation metrics)

## Dataset

The built-in Iris dataset from scikit-learn:

- **Features:** Sepal length, sepal width, petal length, petal width
- **Classes:** Setosa, Versicolor, Virginica
- **Samples:** 150 total, split 80/20 into training and test sets

## How It Works

1. **Split:** 80/20 train-test split with stratified sampling to keep class balance
2. **Scale:** Features standardized with `StandardScaler` so no single measurement dominates the distance calculation
3. **Train:** `KNeighborsClassifier` with `k=5`
4. **Evaluate:** Confusion matrix and full classification report (precision, recall, F1-score) on the held-out test set

## Project Structure

```text
DecodeLabs_Data_Classification/
├── data_classification.py   # Main script
└── README.md                # Project documentation
```

## Getting Started

```bash
pip install scikit-learn
python data_classification.py
```

The script trains the classifier and prints the confusion matrix and classification report to the console.

## What I Learned

KNN is deceptively simple but scaling matters a lot — without it, features with larger numeric ranges quietly dominate the distance calculation and skew every prediction.

## Author

Wireko Fosu Eric — AI Intern, DecodeLabs.
