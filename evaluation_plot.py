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
def plot_confusion_matrix(y_true, y_pred, title="Confusion Matrix"):
    labels = ["yes", "no", "maybe"]
    cm = confusion_matrix(y_true, y_pred)

    print("==== Error Analysis ====")
    for i, true_label in enumerate(labels):
        total = cm[i].sum()
        correct = cm[i][i]
        errors = total - correct
        print(f"\nTrue={true_label} (total={total}, correct={correct}, error={errors}):")
        for j, pred_label in enumerate(labels):
            if i != j and cm[i][j] > 0:
                print(f"   Predicted {pred_label}: {cm[i][j]}/{total} ({cm[i][j]/total*100:.1f}%)")

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix({title})")
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

def get_errors_by_type(data, y_true, y_pred, true_class, pred_class, n=3):
    #LABEL_NAMES_INV = {0: "yes", 1: "no", 2: "maybe"}
    label_map = {"yes": 0, "no": 1, "maybe": 2}

    test_items = list(data.test_set.items())

    cases = [
        (pmid, sample)
        for i, (pmid, sample) in enumerate(test_items)
        if y_true[i] == label_map[true_class] and y_pred[i] == label_map[pred_class]
    ]

    print(f"True={true_class}  Predicted={pred_class}  "
          f"(total {len(cases)} errors, showing {min(n, len(cases))})")

    for pmid, sample in cases[:n]:
        print(f"  Question: {sample['QUESTION']}")
        print(f"  Context: {sample['CONTEXTS']}")

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