"""
Trains the Sentinel linear classifier on extracted residual stream activations.

Usage:
    python sentinel/train_sentinel.py --X_path sentinel_activations_X.npy --y_path sentinel_activations_y.npy --output_model sentinel_model.pkl
"""

import argparse
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--X_path", required=True)
    parser.add_argument("--y_path", required=True)
    parser.add_argument("--output_model", default="sentinel_model.pkl")
    args = parser.parse_args()

    X = np.load(args.X_path)
    y = np.load(args.y_path)

    if len(X.shape) > 2:
        X = X.reshape(X.shape[0], -1)

    sentinel = LogisticRegression(
        C=0.1, class_weight='balanced', solver='lbfgs', max_iter=2000, random_state=42
    )

    print("Running 10-fold cross-validation...")
    cv_scores = cross_val_score(sentinel, X, y, cv=10, scoring='accuracy')
    print(f"Mean CV Accuracy: {cv_scores.mean():.2%} (+/- {cv_scores.std():.4f})")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    sentinel.fit(X_train, y_train)
    y_pred = sentinel.predict(X_test)

    print("\n--- SENTINEL EVALUATION REPORT ---")
    print(classification_report(y_test, y_pred, target_names=['Safe', 'Risk']))

    joblib.dump(sentinel, args.output_model)
    print(f"Sentinel model saved to {args.output_model}")


if __name__ == "__main__":
    main()