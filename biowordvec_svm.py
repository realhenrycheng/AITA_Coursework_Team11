import json
import numpy as np
from gensim.models import KeyedVectors
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix

from pubmed_loader import PubMedQAData
from preprocessing import preprocess_biowordvec, clean_text_plus

# BioWordVec pretrained vectors can be downloaded from:
# https://github.com/ncats/BioSentVec
# File: BioWordVec_PubMed_MIMICIII_d200.vec.bin (word2vec bin format)
MODEL_PATH  = r"E:\NLP_Models\BioWordVec_PubMed_MIMICIII_d200.vec.bin"
LABEL_MAP   = {"yes": 0, "no": 1, "maybe": 2}
LABEL_MAP_INV = {0: "yes", 1: "no", 2: "maybe"}
LABEL_NAMES = ["yes", "no", "maybe"]


def load_biowordvec(model_path):
    print("Loading BioWordVec model...")
    model = KeyedVectors.load_word2vec_format(model_path, binary=True)
    print(f"Model loaded. Vocabulary size: {len(model)}")
    return model


def vectorize(tokens_list, model, dim=200):
    """
    Convert a list of token lists into a matrix of sentence vectors.
    Each sentence vector is the mean of its token vectors.
    Tokens not found in the model vocabulary are skipped.
    If no tokens are found, a zero vector is used.
    """
    vectors = []
    for tokens in tokens_list:
        token_vecs = [model[t] for t in tokens if t in model]
        if token_vecs:
            vectors.append(np.mean(token_vecs, axis=0))
        else:
            vectors.append(np.zeros(dim))
    return np.array(vectors)


def train_svm(X_train, y_train):
    """
    Train SVM with Grid Search + 5-fold cross-validation.
    """
    param_grid = {
        "C":      [0.1, 1, 10],
        "kernel": ["linear", "rbf"],
    }
    grid_search = GridSearchCV(
        SVC(class_weight="balanced"),
        param_grid,
        cv=5,
        scoring="f1_macro",
        verbose=1
    )
    grid_search.fit(X_train, y_train)
    print(f"\nBest parameters: {grid_search.best_params_}")
    print(f"Best CV F1 (macro): {grid_search.best_score_:.4f}")
    return grid_search.best_estimator_


def evaluate_sklearn(clf, X_test, y_test):
    """
    Evaluate using sklearn metrics.
    """
    y_pred = clf.predict(X_test)
    print("\n[sklearn] Classification Report:")
    print(classification_report(y_test, y_pred, target_names=LABEL_NAMES))
    print("[sklearn] Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    return y_pred


def evaluate_official(y_pred, test_pmids, output_path="predictions.json"):
    """
    Save predictions in the format required by the official evaluation.py script.
    Run: python evaluation.py predictions.json
    """
    predictions = {pmid: LABEL_MAP_INV[int(pred)] for pmid, pred in zip(test_pmids, y_pred)}
    with open(output_path, "w") as f:
        json.dump(predictions, f)
    print(f"\n[Official] Predictions saved to {output_path}")
    print("Run: python evaluation.py predictions.json")


if __name__ == "__main__":
    # Load data (relative path, run from repository/ directory)
    data = PubMedQAData("data")

    # Preprocess train and dev
    train_tokens, train_labels = preprocess_biowordvec(data.train_set, clean_fn=clean_text_plus)
    dev_tokens,   dev_labels   = preprocess_biowordvec(data.dev_set,   clean_fn=clean_text_plus)

    # Preprocess test: tokens from preprocess_biowordvec, labels from test_ground_truth
    test_tokens, _ = preprocess_biowordvec(data.test_set, clean_fn=clean_text_plus)
    test_pmids     = list(data.test_set.keys())
    test_labels    = [LABEL_MAP[data.test_ground_truth[pmid]] for pmid in test_pmids]

    # Combine train + dev for cross-validation (500 samples total)
    all_tokens = train_tokens + dev_tokens
    all_labels = train_labels + dev_labels

    # Load BioWordVec
    bwv_model = load_biowordvec(MODEL_PATH)

    # Vectorize
    X_train = vectorize(all_tokens, bwv_model)
    X_test  = vectorize(test_tokens, bwv_model)
    y_train = np.array(all_labels)
    y_test  = np.array(test_labels)

    print(f"\nX_train shape: {X_train.shape}")
    print(f"X_test shape:  {X_test.shape}")

    # Train SVM with Grid Search + 5-fold CV
    svm_clf = train_svm(X_train, y_train)

    # Evaluate with sklearn
    y_pred = evaluate_sklearn(svm_clf, X_test, y_test)

    # Save predictions for official evaluation.py
    evaluate_official(y_pred, test_pmids)