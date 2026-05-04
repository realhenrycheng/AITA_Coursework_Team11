# Team Contributions — Group 11

**EMATM0067 Introduction to AI and Text Analytics**
**Task 1: Medical Question Answering**

---

## Members

| Name | Email | Student ID |
|------|-------|-----------|
| Haoran Cheng | qu25714@bristol.ac.uk | 2801283 |
| Lingshi Gong | st25270@bristol.ac.uk | 2718589 |
| Anis Binti Shahrulhisham | rl25787@bristol.ac.uk | 2800898 |
| Pallavi Mallepura Kadirappa | je25284@bristol.ac.uk | 2747384|
| Alex Verboom | py25014@bristol.ac.uk | |

---

## Section Allocation

### Haoran Cheng

**Project Lead** — responsible for overall project coordination, meeting scheduling, progress tracking, and task allocation across the team.

| Section | Contribution |
|---------|-------------|
| 1. Abstract | Full section |
| 3.1.2 BioWordVec | Axis 1 BioWordVec subsection |
| 3.3 Preprocessing | Full section |
| 4.1 BioGPT | BioGPT zero-shot design and rationale |
| 5.2.3 BioWordVec + SVM | Experimental setup |
| 5.2.4 BioWordVec + MLP | Experimental setup |
| 5.2.7 BioGPT zero-shot | Experimental setup |
| 5.3.1 Result Table | Unified result table  |
| 5.3.3 BioWordVec | Results paragraph |
| 5.3.5 BioGPT Zero-shot | Results paragraph |
| 5.4.2 BioWordVec | Error analysis |
| 5.4.4 BioGPT zero-shot | Error analysis |
| 6.1 Generative vs. discriminative | Discussion |
| 7. Conclusions | Full section |

---

### Lingshi Gong

| Section | Contribution |
|---------|-------------|
| 2.3 Pipeline Overview | Full section |
| 3.1.1 TF-IDF | Axis 1 TF-IDF subsection |
| 3.4 Baseline | Full section |
| 4.2 LoRA | LoRA subsection |
| 5.1 Evaluation Metrics | Full section |
| 5.2.0 TF-IDF + LR (Pilot Experiment) | Experimental setup |
| 5.2.1 TF-IDF + SVM | Experimental setup |
| 5.2.2 TF-IDF + MLP | Experimental setup |
| 5.3.2 TF-IDF | Results paragraph |
| 5.3.6 BioGPT LoRA | Results paragraph |
| 5.4.1 TF-IDF | Error analysis |
| 6.2 LoRA Fine-tuning: Benefits and Limitations | Discussion |

---

### Anis Binti Shahrulhisham

| Section | Contribution |
|---------|-------------|
| 3.1.3 PubMedBERT | Axis 1 PubMedBERT subsection |
| 3.2 Axis 2 Classifiers | Full section (SVM and MLP) |
| 5.2.6 PubMedBERT + MLP | Experimental setup |
| 5.3.4 PubMedBERT | PubMedBERT + MLP results paragraph |
| 5.4.3 PubMedBERT | Error analysis (full) |

---

### Pallavi Mallepura Kadirappa

| Section | Contribution |
|---------|-------------|
| 2.1 Background & Motivation | Full section |
| 5.2.5 PubMedBERT + SVM | Experimental setup |
| 8. Reference | Full section |

---

### Alex Verboom

| Section | Contribution |
|---------|-------------|
| 2.2 Dataset & EDA | Full section |
| 5.5 Answer Quality Prediction | Full section |

---

## Jupyter Notebook (.ipynb)

| File Name                                   | Description                                      | Owner   |
|---------------------------------------------|--------------------------------------------------|---------|
| Answer Quality Prediction.ipynb             | Full pipeline for predicting answer quality      | Alex    |
| EDA.ipynb                                   | Initial Exploratory data analysis                | Lingshi |
| final_eda.ipynb                             | Final EDA                                        | Alex    | 
| baseline.ipynb                              | Baseline model                                   | Lingshi |
| pubmed_loader.ipynb                         | PubMed loader                                    | Haoran  |
| loadData_example.ipynb                      | Data loading example                             | Haoran  |

#### TF-IDF Notebooks
| File Name                                   | Description                                      | Owner   |
|---------------------------------------------|--------------------------------------------------|---------|
| TFIDF_MLP.ipynb                             | TF‑IDF vectorisation + MLP model                 | Lingshi |
| TFIDF_SVM.ipynb                             | TF‑IDF vectorisation + SVM model                 | Lingshi |

#### PubMedBERT Notebooks
| File Name                                   | Description                                      | Owner   |
|---------------------------------------------|--------------------------------------------------|---------|
| pubmedbert_mlp.ipynb                        | PubMedBERT embeddings + MLP for comparison       | Anis    |
| pubmedbert_mlp_original.ipynb               | PubMedBERT MLP Pipeline without class weight     | Anis    |
| pubmedbert_mlp_improved.ipynb               | Improved architecture, tuning, and evaluation    | Anis    |
| PubMedBERT_SVM(1) (1).ipynb                 | PubMedBERT embeddings + SVM for comparison       | Pallavi |

#### BioWordVec Notebooks
| File Name                                   | Description                                      | Owner   |
|---------------------------------------------|--------------------------------------------------|---------|
| biowordvec_mlp.ipynb                        | BioWordVec embeddings + MLP classifier           | Haoran  |
| biowordvec_svm.ipynb                        | BioWordVec embeddings + SVM classifier           | Haoran  |

#### BioGPT Notebooks
| File Name                                   | Description                                      | Owner   |
|---------------------------------------------|--------------------------------------------------|---------|
| biogpt_zeroshot_experiment.ipynb            | BioGPT zero‑shot experiment                      | Haoran  |
| biogpt_prompt_test_experiment.ipynb         | BioGPT prompt testing                            | Haoran  |
| biogpt_lora_experiment.ipynb                | BioGPT LoRA experiment                           | Lingshi |
| biogpt_lora_finetune.ipynb                  | BioGPT LoRA fine‑tuning                          | Lingshi |
| biogpt_lora_finetune_roleprompt.ipynb       | BioGPT LoRA + role prompt                        | Lingshi |
| biogpt_lora_roleprompt_experiment.ipynb     | BioGPT role‑prompt LoRA experiment               | Lingshi |
| biogpt_optimization_EN.ipynb                | BioGPT optimization (EN)                         | Haoran  |
| biogpt_optimization_lora.ipynb              | BioGPT LoRA optimization                         | Lingshi |
| biogpt_optimization_lora_origin.ipynb       | BioGPT LoRA (control/original)                   | Lingshi |

## Python Script (.py)

| File Name                   | Description                                            | Owner   |
|-----------------------------|--------------------------------------------------------|---------|
| evaluation.py               | Core evaluation utilities                              | Lingshi |
| evaluation_plot.py          | Function for generating evaluation plots               | Lingshi |
| preprocessing.py            | Text preprocessing functions (cleaning)                | Lingshi |
| pubmedbert_preprocess.py    | Preprocessing specifically for PubMedBERT + MLP models | Anis    |
| pubmed_loader.py            | Load and structured PubMed dataset for modelling       | Haoran  |
| get_human_performance.py    | Compute human baseline performance for comparison      | Haoran  |


*Last updated: May 2026*
