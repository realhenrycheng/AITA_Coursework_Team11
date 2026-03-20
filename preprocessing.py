import os
from pubmed_loader import PubMedQAData
from transformers import AutoTokenizer
import torch

LABEL_MAP = {"yes": 0, "no": 1, "maybe": 2}


def build_input(sample, use_mesh=False):
    question = sample["QUESTION"]
    context  = " ".join(sample["CONTEXTS"])
    if use_mesh:
        context += " " + " ".join(sample["MESHES"])
    return question, context


def pubmedbert_preprocess(dataset, tokenizer, labels_source=None, use_mesh=False, max_length=512):
    questions, contexts, labels = [], [], []

    for pmid, sample in dataset.items():
        question, context = build_input(sample, use_mesh)
        questions.append(question)
        contexts.append(context)
        label = labels_source[pmid] if labels_source else sample["final_decision"]
        labels.append(LABEL_MAP[label])

    encodings = tokenizer(
        questions,
        contexts,
        max_length=max_length,
        truncation="only_second",
        padding="max_length",
        return_tensors="pt"
    )

    return encodings, torch.tensor(labels)

# from bert_preprocessing import preprocess

# train_encodings, train_labels = preprocess(data.train_set, tokenizer)
# dev_encodings,   dev_labels   = preprocess(data.dev_set,   tokenizer)
# test_encodings,  test_labels  = preprocess(data.test_set,  tokenizer, 
#                                            labels_source=data.test_ground_truth)


# do the biowordvec preprocessing

import re
import string
from pubmed_loader import PubMedQAData

LABEL_MAP = {"yes": 0, "no": 1, "maybe": 2}

def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = text.split()
    return tokens

def build_biowordvec_input(sample):
    question = sample["QUESTION"]
    context  = " ".join(sample["CONTEXTS"])
    return question + " " + context

def preprocess_biowordvec(dataset, labels_source=None):
    tokens_list = []
    labels      = []

    for pmid, sample in dataset.items():
        text   = build_biowordvec_input(sample)
        tokens = clean_text(text)
        tokens_list.append(tokens)
        label = labels_source[pmid] if labels_source else sample["final_decision"]
        labels.append(LABEL_MAP[label])

    return tokens_list, labels


# now do the TF-IDF preprocessing

def build_tfidf_input(sample, use_mesh=False):
    question = sample["QUESTION"]
    context  = " ".join(sample["CONTEXTS"])
    text     = question + " " + context
    if use_mesh:
        text += " " + " ".join(sample["MESHES"])
    return text


def preprocess_tfidf(dataset, labels_source=None, use_mesh=False):
    texts  = []
    labels = []

    for pmid, sample in dataset.items():
        text = build_tfidf_input(sample, use_mesh)
        texts.append(text)
        label = labels_source[pmid] if labels_source else sample["final_decision"]
        labels.append(LABEL_MAP[label])

    return texts, labels