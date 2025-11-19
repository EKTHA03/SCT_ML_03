# python
"""
Train a linear SVM to classify cats vs dogs images.

Expect dataset layout (examples):
 - ./train/cat.0.jpg
 - ./train/dog.0.jpg
Or folders named 'cat'/'dog' or 'PetImages/Cat' and 'PetImages/Dog'.

If no data is found the script will auto-generate a small synthetic dataset
in the provided --data_dir so you can run the script end-to-end.
"""
import os
import argparse
import sys
from collections import Counter

# Friendly dependency checks with actionable install hints
try:
    from PIL import Image
except Exception:
    print("Missing dependency: Pillow. Install with:")
    print(f'  "{sys.executable}" -m pip install --upgrade Pillow')
    sys.exit(1)

try:
    import numpy as np
except Exception:
    print("Missing dependency: numpy. Install with:")
    print(f'  "{sys.executable}" -m pip install --upgrade numpy')
    sys.exit(1)

try:
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import LinearSVC
except Exception:
    print("Missing dependency: scikit-learn. Install with:")
    print(f'  "{sys.executable}" -m pip install --upgrade scikit-learn')
    sys.exit(1)

try:
    from joblib import dump
except Exception:
    print("Missing dependency: joblib. Install with:")
    print(f'  "{sys.executable}" -m pip install --upgrade joblib')
    sys.exit(1)

IMAGE_SIZE = (128, 128)  # (width, height)
RANDOM_STATE = 42
ALLOWED_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif')


def find_cat_dog_files_recursive(root_dir):
    cats = []
    dogs = []
    if not os.path.exists(root_dir):
        return cats, dogs
    for root, _, files in os.walk(root_dir):
        parent = os.path.basename(root).lower()
        for f in files:
            lf = f.lower()
            name, ext = os.path.splitext(lf)
            if ext not in ALLOWED_EXTS:
                continue
            fp = os.path.join(root, f)
            if name.startswith('cat') or parent.startswith('cat'):
                cats.append(fp)
            elif name.startswith('dog') or parent.startswith('dog'):
                dogs.append(fp)
    return sorted(cats), sorted(dogs)


def generate_sample_images(target_dir, cats=12, dogs=12):
    os.makedirs(target_dir, exist_ok=True)
    w, h = IMAGE_SIZE
    for i in range(cats):
        rr = (np.arange(h).reshape(h, 1) + i * 3) % 256
        cc = (np.arange(w).reshape(1, w) * 2 + i * 5) % 256
        img = ((rr + cc) // 2).astype(np.uint8)
        Image.fromarray(img).save(os.path.join(target_dir, f"cat.{i}.jpg"))
    for i in range(dogs):
        rr = (np.arange(h).reshape(h, 1) * (i + 1)) % 256
        cc = (np.arange(w).reshape(1, w) + i * 7) % 256
        img = (rr ^ cc).astype(np.uint8)
        Image.fromarray(img).save(os.path.join(target_dir, f"dog.{i}.jpg"))


def load_dataset(data_dir, auto_generate=True):
    if data_dir and os.path.exists(data_dir):
        cats, dogs = find_cat_dog_files_recursive(data_dir)
        if cats and dogs:
            return cats, dogs

    cwd = os.getcwd()
    candidates = [
        data_dir,
        os.path.join(cwd, 'train'),
        os.path.join(cwd, 'Train'),
        os.path.join(cwd, 'PetImages'),
        cwd,
    ]
    for c in candidates:
        if not c:
            continue
        cats, dogs = find_cat_dog_files_recursive(c)
        if cats and dogs:
            return cats, dogs

    if auto_generate:
        target = data_dir if data_dir else os.path.join(os.getcwd(), "train")
        print(f"No cat/dog images found. Generating a small synthetic dataset at: {os.path.abspath(target)}")
        generate_sample_images(target, cats=24, dogs=24)
        cats, dogs = find_cat_dog_files_recursive(target)
        if cats and dogs:
            return cats, dogs

    return [], []


def preprocess_and_vectorize(file_list, verbose=False):
    X = []
    dims = []
    for i, fp in enumerate(file_list):
        try:
            with Image.open(fp) as im:
                orig_size = im.size  # (width, height)
                dims.append(orig_size)
                im = im.convert('L')
                im = im.resize(IMAGE_SIZE, Image.BILINEAR)
                arr = np.asarray(im, dtype=np.float32) / 255.0
                X.append(arr.flatten())
        except Exception as e:
            dims.append(None)
            if verbose:
                print(f"Warning: failed to read {fp}: {e}")
    if not X:
        return np.zeros((0, IMAGE_SIZE[0] * IMAGE_SIZE[1]), dtype=np.float32), dims
    return np.stack(X, axis=0), dims


def summarize_dims(dims):
    valid = [d for d in dims if d is not None]
    if not valid:
        return "none"
    widths = [d[0] for d in valid]
    heights = [d[1] for d in valid]
    mean_w = int(np.mean(widths))
    mean_h = int(np.mean(heights))
    min_w, max_w = min(widths), max(widths)
    min_h, max_h = min(heights), max(heights)
    common = Counter(valid).most_common(5)
    common_str = ", ".join([f"{w}x{h}({c})" for (w, h), c in common])
    return f"mean {mean_w}x{mean_h}, range {min_w}x{min_h} - {max_w}x{max_h}; top sizes: {common_str}"


def main():
    parser = argparse.ArgumentParser(description="Train linear SVM on Cats vs Dogs (128x128 grayscale).")
    parser.add_argument("--data_dir", type=str, default="./train", help="Path to folder with images (or where sample will be generated)")
    parser.add_argument("--model_out", type=str, default="svm_cats_dogs.joblib", help="Output path for saved model")
    parser.add_argument("--no_auto_generate", action="store_true", help="Do not auto-generate sample images if none found")
    parser.add_argument("--show_dims", action="store_true", help="Print per-image spatial dimensions (may be verbose)")
    parser.add_argument("--verbose", action="store_true", help="Verbose messages while loading images")
    args = parser.parse_args()

    cats, dogs = load_dataset(args.data_dir, auto_generate=not args.no_auto_generate)
    if not (cats and dogs):
        print("ERROR: No images found. Looked in:", os.path.abspath(args.data_dir))
        print("Ensure your train folder contains files named like 'cat.0.jpg' and 'dog.0.jpg',")
        print("or subfolders 'cat'/'dog' or 'PetImages/Cat' and 'PetImages/Dog'.")
        print("You can allow the script to auto-generate a small sample by omitting --no_auto_generate.")
        sys.exit(1)

    print(f"Found {len(cats)} cat images and {len(dogs)} dog images.")

    X_cats, dims_cats = preprocess_and_vectorize(cats, verbose=args.verbose)
    X_dogs, dims_dogs = preprocess_and_vectorize(dogs, verbose=args.verbose)
    if X_cats.shape[0] == 0 or X_dogs.shape[0] == 0:
        print("ERROR: No readable images after preprocessing. Check earlier messages.")
        sys.exit(1)

    print("\nSpatial dimensions summary (original image sizes):")
    print(f"  Cats: {summarize_dims(dims_cats)}")
    print(f"  Dogs: {summarize_dims(dims_dogs)}")
    if args.show_dims:
        print("\n  First 20 cat image sizes:")
        for i, d in enumerate(dims_cats[:20], 1):
            size_str = f"{d[0]}x{d[1]}" if d else "unreadable"
            print(f"    {i:2d}. {size_str}  -> {cats[i-1]}")
        print("\n  First 20 dog image sizes:")
        for i, d in enumerate(dims_dogs[:20], 1):
            size_str = f"{d[0]}x{d[1]}" if d else "unreadable"
            print(f"    {i:2d}. {size_str}  -> {dogs[i-1]}")

    X = np.vstack([X_cats, X_dogs])
    y = np.hstack([np.zeros(X_cats.shape[0], dtype=np.int64), np.ones(X_dogs.shape[0], dtype=np.int64)])

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_STATE
        )

    print(f"\nTraining samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

    clf = make_pipeline(StandardScaler(), LinearSVC(random_state=RANDOM_STATE, max_iter=20000))
    print("Training SVM...")
    clf.fit(X_train, y_train)
    print("Training complete.")

    if X_test.shape[0] > 0:
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

    dump(clf, args.model_out)
    print(f"Saved trained model to: {os.path.abspath(args.model_out)}")


if __name__ == "__main__":
    main()