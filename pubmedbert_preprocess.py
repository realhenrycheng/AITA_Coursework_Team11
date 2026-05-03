import torch
import os
from dotenv import load_dotenv
from huggingface_hub import login
from transformers import AutoTokenizer

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

load_dotenv()  # loads .env file
login(token=os.getenv("HF_TOKEN"))

MODEL_NAME = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
MAX_LEN = 512

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

LABEL_MAP = {
    "yes": 0,
    "no": 1,
    "maybe": 2
}

# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------

def build_input(sample, use_mesh=False):
    question = sample.get("QUESTION", "")
    context_list = sample.get("CONTEXTS", [])
    context = " ".join(context_list)

    if use_mesh:
        mesh_list = sample.get("MESHES", [])
        if mesh_list:
            # PREPEND MeSH, do NOT append or duplicate context
            context = "MeSH: " + "; ".join(mesh_list) + " " + context

    return question, context



def get_label(sample):
    """
    Extract the final decision label from a PubMedQA sample.
    """
    return sample["final_decision"]


# ---------------------------------------------------------
# Main preprocessing function
# ---------------------------------------------------------

def pubmedbert_preprocess(dataset, tokenizer, use_mesh=False, max_length=512):
    """
    Convert PubMedQA samples into tokenized tensors for PubMedBERT.
    dataset: dict of pmid -> sample
    """
    questions, contexts, labels = [], [], []

    for pmid, sample in dataset.items():
        question, context = build_input(sample, use_mesh)
        questions.append(question)
        contexts.append(context)

        label = get_label(sample)
        labels.append(LABEL_MAP[label])

    encodings = tokenizer(
        questions,
        contexts,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_tensors="pt"
    )

    return encodings, torch.tensor(labels)
    

# ---------------------------------------------------------
# Optional test block
# ---------------------------------------------------------

if __name__ == "__main__":
    print("PubMedBERT preprocessing module loaded successfully.")