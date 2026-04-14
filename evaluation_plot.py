from sklearn.metrics import accuracy_score, f1_score, classification_report
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

# Import the following functions by add--
# from evaluation_plot import evaluate, plot_confusion_matrix, get_error_samples, training_curves

# Use the func to calculate accuracy_score, f1_score and classification_report
# 'split_name' is the title of the table
def evaluate(y_pred, y_true, split_name):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    print(f"\n── {split_name} ──")
    print(f"Accuracy: {acc:.4f}  Macro-F1: {f1:.4f}")
    print(classification_report(
        y_true, y_pred,
        target_names=["yes", "no", "maybe"]
    ))


# Plot confusion_matrix
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["yes", "no", "maybe"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()

# get error samples
# data = PubMedQAData("data")
# n: the number of samples printed
def get_error_samples(data, y_true, y_pred, n=5):
    print(f"\nError samples (top {n})")
    test_items = list(data.test_set.items())
    errors = [
        (pmid, sample, y_true[i], y_pred[i])
        for i, (pmid, sample) in enumerate(test_items)
        if y_true[i] != y_pred[i]
    ]
    
    LABEL_NAMES_INV = {0: "yes", 1: "no", 2: "maybe"} 
    for pmid, sample, true, pred in errors[:n]:
        print(f"\nPMID: {pmid}")
        print(f"Question : {sample['QUESTION']}")
        print(f"True     : {LABEL_NAMES_INV[true]}")
        print(f"Predicted: {LABEL_NAMES_INV[pred]}")

# plot loss curves
def training_curves(train_losses, dev_losses):
    plt.figure()
    plt.plot(train_losses, label='Training Loss')
    plt.plot(dev_losses, label='Validation Loss')

    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.title("Training curves")
    plt.legend()
    plt.show()