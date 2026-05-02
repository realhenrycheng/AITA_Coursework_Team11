import torch
import string
import re
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer

LABEL_MAP = {"yes": 0, "no": 1, "maybe": 2}


def build_input(sample, use_mesh=False):
    question = sample["QUESTION"]
    context  = " ".join(sample["CONTEXTS"])
    if use_mesh:
        context += " " + " ".join(sample["MESHES"])
    return question, context

def get_label(sample):
    return sample["final_decision"].lower().strip()

# pubmedbert preprocessing
def pubmedbert_preprocess(dataset, tokenizer, use_mesh=False, max_length=512):
    questions, contexts, labels = [], [], []

    for pmid, sample in dataset.items():
        question, context = build_input(sample, use_mesh)
        questions.append(question)
        contexts.append(context)
        # label = labels_source[pmid] if labels_source else sample["final_decision"]
        label = get_label(sample)
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

'''
# error analysis
def clean_text_sim_sw(text):
    result=[]
    for token in simple_preprocess(text):
        if token not in STOPWORDS:
            result.append(WordNetLemmatizer().lemmatize(token, 'v'))
    return result
'''

def clean_text_basic(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation)) # Delete all punctuation
    tokens = text.split()
    return tokens

# Not sure if this is better, more research needed.
def clean_text_plus(text):
    text = text.lower()

    # Keep the hyphen and remove the rest of the punctuation
    keep = {"-"}
    puncts = "".join(c for c in string.punctuation if c not in keep)
    text = text.translate(str.maketrans("", "", puncts))

    # Delete purely numeric tokens (keep alphabetic ones such as "il-6")
    tokens = [t for t in text.split() if not re.fullmatch(r"[\d.]+", t)]
    return tokens

'''
def build_biowordvec_input(sample):
    question = sample["QUESTION"]
    context  = " ".join(sample["CONTEXTS"])
    return question + " " + context
'''

# biowordvec preprocessing
def preprocess_biowordvec(dataset, clean_fn=clean_text_plus, use_mesh=False):
    tokens_list = []
    labels      = []

    for pmid, sample in dataset.items():
        question, context = build_input(sample, use_mesh)
        tokens = clean_fn(question + " " + context)
        tokens_list.append(tokens)
        #label = labels_source[pmid] if labels_source else sample["final_decision"]
        label = get_label(sample)
        labels.append(LABEL_MAP[label])

    return tokens_list, labels # List of token list + integers list


# TF-IDF preprocessing
'''
def build_tfidf_input(sample, use_mesh=False):
    question = sample["QUESTION"]
    context  = " ".join(sample["CONTEXTS"])
    text     = question + " " + context
    if use_mesh:
        text += " " + " ".join(sample["MESHES"])
    return text
'''

def preprocess_tfidf(dataset, use_mesh=False):
    texts  = []
    labels = []

    for pmid, sample in dataset.items():
        question, context = build_input(sample, use_mesh)
        texts.append(question + " " + context)
        # label = labels_source[pmid] if labels_source else sample["final_decision"]
        label = get_label(sample)
        labels.append(LABEL_MAP[label])

    return texts, labels # String list + integer list