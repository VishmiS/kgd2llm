# Enhancing Semantic Search with Knowledge Graph-Guided D2LLMs

A hybrid **semantic search framework** that combines **Large Language Models (LLMs)** with **Knowledge Graphs (KGs)** to improve retrieval accuracy, interpretability, and efficiency.  

This framework integrates structured knowledge into semantic search and distills LLM capabilities into an efficient model for large-scale retrieval.

---

## 🚀 Purpose

This project is designed to:

- Enable **knowledge-aware semantic search** and question answering  
- Improve **factual accuracy** and **query disambiguation** using Knowledge Graphs  
- Provide a **computationally efficient** solution via LLM distillation  
- Serve as a research-ready framework for IR (Information Retrieval) tasks and hybrid AI applications  

Key advantages:

- KG-guided query enrichment improves recall and precision  
- Teacher–student distillation transfers LLM knowledge into lightweight models  
- Hybrid neural-symbolic design allows **efficient and interpretable retrieval**  

---

## 📦 Installation & Setup

1. Clone this repository:  
```bash
git clone <repo_url>
cd <repo_folder>
```

2. Install dependencies (Python 3.10+ recommended):
```bash
pip install -r requirements.txt
```

3. Prepare datasets (see Datasets below) and place them in the designated directories.
4. Prepare teacher logits and hard negatives for distillation (if not available, they will be generated automatically).

## 🗂️ Code Structure

```text
├── baselines/              # Baseline models for comparison
├── datasets/               # Dataset files (JSON,TSV, Pickle)
├── entity_linking/         # Entity linking code
├── evaluate/               # Evaluation scripts
├── model/                  # D2LLM model implementation
├── outputs/                # place where all .pkl files are stored after preprocessing
├── preprocess/             # Preprocssing for features, positives and  hard negatives
├── results/                # Evaluation results for each dataset
├── utils/                  # Helpers for model saving, logging, and debugging
├── train.py                # Training code
├── data_loader.py          # Dataloader for training
├── loss.py                 # Loss functions
└── train<datasetname>.sh   # Trigger training process for a dataset

```

## 🧩 Core Components

### 1. Teacher–Student Framework *(from Liao et al., 2024)*

> The following design is based on the **D2LLM architecture** proposed in [Liao et al., 2024](https://aclanthology.org/2024.acl-long.791/):

- **Teacher Model (Cross-Encoder):** High-precision query–document ranking  
- **Student Model (Bi-Encoder):** Efficient retrieval via independent embeddings  
- **Knowledge Distillation:** Transfers semantic reasoning and ranking behavior from teacher → student  

> This framework allows us to leverage the **nuanced understanding of a large LLM** while keeping retrieval efficient for large-scale datasets.

### 2. Interaction Modules *(from Liao et al., 2024)*

- **PMA (Pooling by Multihead Attention):** Aggregates token representations flexibly for downstream retrieval  
- **IEM (Interaction Emulation Module):** Mimics the cross-attention behavior of the teacher model to improve student model performance  

> These modules are key to **decomposing cross-encoder reasoning** into a lightweight bi-encoder, enabling efficient yet accurate semantic search.

---

## 🔍 Query Enrichment Pipeline

### Knowledge Sources

- Wikidata  
- DBpedia  

### Features

- Entity linking (**Falcon 2.0**, fallback: DBpedia Spotlight)  
- Relation extraction  
- Fact retrieval via SPARQL queries  
- Semantic similarity filtering  
- Natural language fact generation  

### Composite Label Detection & Phrase Decomposition

- Identifies complex relational phrases using linguistic cues (prepositions, verbs, phrase length)  
- Decomposes phrases into structured `(property, entity)` pairs using rule-based parsing and pattern matching  
- Resolves and enriches these components via Wikidata/DBpedia queries and similarity-based ranking  

### Question Intent Awareness & Semantic Filtering

- Classifies queries into intent categories (e.g., location, time, person)  
- Uses intent signals to prioritize relevant properties and refine entity–property mappings  
- Applies semantic filtering and similarity scoring to extract precise, context-aware answers from knowledge graphs  

---

## 📂 Datasets

Supported datasets for training and evaluation:

| Dataset      | Task                  | Notes                          |
|--------------|---------------------|--------------------------------|
| MS MARCO     | Web Search           | Query–document pairs           |
| WebQuestions | Open-Domain QA       | Question-answer retrieval      |
| TREC-COVID   | Biomedical retrieval | Domain-specific IR             |

### 📂 Required Files for IR Training

#### 1. Dataset TSV
- Columns:  
  - `sentence1` → query  
  - `sentence2` → relevant document  

#### 2. Positive Logits (Pickle)
- Path: `pos_dir/pos_emb/*.pkl`  
- Format:  
```json
{ (query, document): [logit1, logit2] }
```

If missing, logits are automatically generated using a teacher model.
 
#### 3. Hard Negatives (Pickle)
- Path: ` neg_dir/logits/*.pkl` 
- Format:
```json
{
  "query": [
    (negative_doc, [logit1, logit2]),
    ...
  ]
}
```
Used to sample K hard negatives per query during training.


## 🏋️ Training

To start training the model, use the provided shell scripts. For each dataset, a dedicated script is created for convenience:
```bash
scripts/train<datasetname>.sh
```

### What the script does

- Each script sets all hyperparameters, paths, and distributed training settings. 
- It triggers train.py with the correct dataset paths, batch size, negative sampling, logits, and model configurations. 
- Update paths like TB_DIR, TRAINING_LOG, and OUTPUT_DIR before running. 


- Loads dataset, positive logits, and hard negatives  
- Configures model, batch size, negative sampling, and optimizer  
- Trains the student model via knowledge distillation  

⚠️ Make sure the required dataset files, positive logits, and hard negatives are prepared as described in the dataloader section.
The preprocess files are prepared to help create the files required for the dataloader.
Start with creating the hard negatives, positives and then features.

### Training Details

- **Optimizer:** AdamW  
- **Mixed precision:** bfloat16  
- **Loss:** Contrastive, hard-negative, ranking distillation, feature alignment  
- **Scheduler:** Warmup + cosine decay  
- **Techniques:** Adaptive loss, dynamic temperature, embedding regularization, gradient clipping  

### Outputs

- **Model checkpoints:** `{output_dir}/`  
- **TensorBoard logs:** `{tb_dir}/`  
- **Training logs:** `{training_log}/`  

---

## ✅ Evaluation

- Evaluation scripts are in `evaluate/`  
- Baselines are in `baselines/`  
- Metrics: Recall@10, MRR, ranking performance  

---

## ⚠️ Limitations

- Errors in entity linking may cause semantic drift  
- KG alignment must be context-aware  
- Retrieval depends on the quality of KG facts  