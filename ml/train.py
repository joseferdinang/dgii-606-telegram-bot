"""Training script for the invoice field classifier with enhanced Dominican dataset."""
from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from ml.feature_engineering import tokenize, build_features
from ml.synthetic_generator import generate_dataset


def main():
    print("Generating comprehensive synthetic dataset (175+ Dominican company profiles, fuel, hardware, tech)...")
    texts, labels_list = generate_dataset(n=3000)
    
    print("Building feature matrices from tokens...")
    X_list = []
    y_list = []
    skipped = 0
    
    for text, labels in zip(texts, labels_list):
        tokens = tokenize(text)
        if len(tokens) != len(labels):
            skipped += 1
            continue
        df = build_features(tokens)
        X_list.append(df)
        y_list.extend([l["label"] for l in labels])
        
    print(f"Tokens processed: {len(y_list)} across {len(X_list)} invoices (skipped: {skipped})")
    X = pd.concat(X_list, ignore_index=True)
    y = y_list
    
    print(f"Dataset shape: {X.shape}")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print("Training HistGradientBoostingClassifier on Dominican invoice dataset...")
    clf = HistGradientBoostingClassifier(
        max_iter=120,
        learning_rate=0.1,
        max_leaf_nodes=31,
        random_state=42,
    )
    clf.fit(X_train, y_train)
    
    print("Evaluating model performance on test set...")
    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred, digits=4)
    print("\nClassification Report:")
    print(report)
    
    model_dir = Path(__file__).parent / "models"
    model_dir.mkdir(exist_ok=True, parents=True)
    model_path = model_dir / "field_classifier.joblib"
    
    joblib.dump(clf, model_path)
    print(f"\nTrained model successfully saved to: {model_path}")


if __name__ == "__main__":
    main()
