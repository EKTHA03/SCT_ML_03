# python
"""
SVM classifier for Kaggle Dogs vs Cats dataset.
Assumes dataset directory structure:
  <dataset_dir>/train/cat.0.jpg
  <dataset_dir>/train/dog.0.jpg

Usage (example):
  python train_svm_cats_dogs.py --data_dir ./train --model_out svm_cats_dogs.joblib
"""

import os
import glob
import argparse
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from joblib import dump

IMAGE_SIZE = (128, 128)  # width, height
RANDOM_STATE = 42


def load_dataset(data_dir):
    """
    Load images from data_dir. Expects files named like cat.*.jpg and dog.*.jpg
    Returns:
      X: numpy array of shape (n_samples, 128*128), dtype=float32, values in [0,1]
      y: numpy array of shape (n_samples,), labels 0=cat, 1=dog
    """
    X = []
    y = []

    cat_pattern = os.path.join(data_dir, 'cat.*.jpg')
    dog_pattern = os.path.join(data_dir, 'dog.*.jpg')

    cat_files = glob.glob(cat_pattern)
    dog_files = glob.glob(dog_pattern)

    if not cat_files and not dog_files:
        # try more general patterns if exact naming differs
        cat_files = glob.glob(os.path.join(data_dir, 'cat.*'))
        dog_files = glob.glob(os.path.join(data_dir, 'dog.*'))

    # helper to process files
    def process_files(file_list, label):
        for fp in file_list:
            try:
                with Image.open(fp) as img:
                    img = img.convert('L')  # grayscale
                    img = img.resize(IMAGE_SIZE, Image.BILINEAR)
                    arr = np.asarray(img, dtype=np.float32) / 255.0
                    X.append(arr.flatten())
                    y.append(label)
            except Exception as e:
                print(f"Warning: failed to process {fp}: {e}")

    process_files(cat_files, 0)
    process_files(dog_files, 1)

    if len(X) == 0:
        raise RuntimeError(f"No images found in {data_dir}. Confirm files named like 'cat.0.jpg' and 'dog.0.jpg'.")

    X = np.stack(X, axis=0)
    y = np.array(y, dtype=np.int64)
    return X, y


def main(args):
    data_dir = args.data_dir
    model_out = args.model_out

    print("Loading dataset from:", data_dir)
    X, y = load_dataset(data_dir)
    print(f"Loaded {X.shape[0]} samples. Feature dim: {X.shape[1]}")

    # Train/test split 80/20, stratified
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    print(f"Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

    # Train SVM classifier with linear kernel
    clf = SVC(kernel="linear", random_state=RANDOM_STATE)
    print("Training SVM (this may take a while)...")
    clf.fit(X_train, y_train)
    print("Training complete.")

    # Evaluate
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred, target_names=['cat', 'dog'])

    print("\nEvaluation results:")
    print(f"Accuracy: {acc:.4f}")
    print("Confusion Matrix:")
    print(cm)
    print("Classification Report:")
    print(cr)

    # Save model
    dump(clf, model_out)
    print(f"Saved trained model to: {model_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train linear SVM on Cats vs Dogs (grayscale 128x128).")
    parser.add_argument("--data_dir", type=str, default="./train", help="Path to training folder containing cat.*.jpg and dog.*.jpg")
    parser.add_argument("--model_out", type=str, default="svm_cats_dogs.joblib", help="Output path for saved SVM model (.joblib)")
    args = parser.parse_args()
    main(args)