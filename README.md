# 🗂️ ComplaintSense: Intelligent Complaint Classification Using Localised Transformer Architectures

## 🌍 Project Overview
This project applies **transformer-based deep learning** to automatically classify consumer support complaints into one of ten predefined categories, reducing manual triage time and improving routing accuracy for customer service teams. The model was developed and submitted as part of the **ComplaintSense Kaggle competition**, and forms our team's capstone project for the AI Saturdays Lagos cohort.

Traditional keyword and rule-based routing systems struggle with overlapping vocabulary and subtle contextual differences between complaint types (e.g. distinguishing a billing dispute from a subscription cancellation when both mention being "charged"). This project fine-tunes a pretrained transformer to capture that context directly from complaint text.

---

## 📊 Dataset
**Source:** [ComplaintSense: Consumer Complaint Classification Challenge](https://www.kaggle.com/competitions/complaint-sense-consumer-complaint-classification-challenge) (Kaggle)

The dataset consists of short consumer complaint text passages, provided directly by the competition organisers:
- `train_complaints.csv` — 380 labelled examples
- `test_complaints.csv` — 160 unlabeled examples (used for final prediction/submission)

Each complaint is labelled with one of ten categories: `billing`, `product_defect`, `delivery_shipping`, `refund_return`, `customer_service`, `account_access`, `fraud_unauthorized`, `warranty_repair`, `subscription_cancel`, `general_inquiry`.

The exact labelling methodology used by the competition organisers was not disclosed. Given the dataset's small size, class distribution is not perfectly balanced across categories, which directly informed our modelling choices (see Training Pipeline).

We explored supplementing the dataset with external complaint sources and synthetic data augmentation (LLM-based paraphrasing of existing labelled examples), but ultimately did not use augmented data in the final submitted model, due to time constraints. Full reasoning for this decision is documented in our Data Card (`docs/data_card.pdf`).

---

## ⚙️ Training Pipeline

### Preprocessing
- Text was tokenised using the `distilbert-base-uncased` tokeniser (Hugging Face `transformers`), with truncation at a maximum sequence length of 256 tokens.
- Category labels were encoded into integer IDs using a `label2id`/`id2label` mapping built from the training data.
- Since the competition provided only one labelled file, we created our own train/validation split (85%/15%) from `train_complaints.csv`, since no separate labelled validation set was provided.

### Model
- **Base model:** `distilbert-base-uncased`, fine-tuned via `AutoModelForSequenceClassification` with `num_labels=10`.
- **Why DistilBERT:** chosen for its balance of performance and computational efficiency, appropriate for a small dataset and limited compute/timeline, while still leveraging transformer self-attention to capture contextual meaning beyond keyword matching.

### Key design choices
- **Evaluation metric:** Macro-averaged F1 (Balanced F1), matching the competition's own scoring metric, since it weights all ten categories equally regardless of size — important given the dataset's imbalance.
- **Training configuration:** learning rate `2e-5`, batch size `16`, `5 epochs`, with `load_best_model_at_end=True` (selecting the checkpoint with the best validation F1, not simply the final epoch) to reduce overfitting risk on a small dataset.

### Hyperparameter search
No formal search (grid/random) was conducted given timeline constraints. We used standard DistilBERT defaults, adjusting only epoch count (2 → 5) for our small dataset. Systematic tuning is noted as future work.
---

## 📈 Evaluation
The model was evaluated on our held-out validation split (57 examples, ~15% of the training data) using accuracy and macro-averaged F1.

Local validation results (committed run):

Accuracy: 0.9298
Macro F1: 0.8976

Leaderboard result (hidden test set): Macro F1 0.4413 — only marginally above the provided baseline (~0.419).

We observed a substantial gap between local validation and leaderboard performance, most likely due to overfitting on a very small validation set (57 examples, one category with just 1 example), making local F1 an unreliable generalisation estimate: a known risk with small datasets. We could not fully diagnose this before submission closed; a possible contributing factor we didn't have time to rule out is an implementation issue in the label-mapping/inference pipeline. Future work: a larger/stratified validation split, k-fold cross-validation, and closer verification of inference code against training-time label mappings.

This gap underscores a challenge our problem statement identified from the outset — working with limited, imbalanced data, reinforcing the need for more robust evaluation methodology going forward.
**Final validation results:**
- Accuracy: `0.9298 (≈93.0%)`
- Macro F1: `0.8976 (≈89.8%)`
- Trained for 5 epochs, evaluated on a 57-example held-out validation split (15% of training data)

This exceeds the competition-provided baseline (character TF-IDF + logistic regression, ~0.44 Balanced F1).

**Limitations acknowledged:**
- Our validation set is small (57 examples across 10 categories), meaning per-category F1 for rare categories (some with as few as 1 example in the validation split) is statistically noisy and should be interpreted cautiously.
- We manually reviewed model errors on "confusable pairs" (e.g. billing vs. subscription_cancel) flagged by the competition organisers as a known challenge.

---

## 🔁 Reproduction

This project was developed and run in a **Kaggle Notebook** environment, with the competition dataset mounted automatically. To reproduce:

1. **Clone the repository:**
   `git clone https://github.com/starokoronkwo/C10-team-addo.git`

2. **Environment:** Run on Kaggle (recommended) — open `scripts/complaintsense_pipeline.ipynb` as a new Kaggle Notebook attached to the [ComplaintSense competition data](https://www.kaggle.com/competitions/complaint-sense-consumer-complaint-classification-challenge), or install dependencies locally:
   `pip install -r requirements.txt`

3. **Run order:** The notebook is structured to run top-to-bottom as a single pipeline:
   - Data loading & inspection
   - Tokenization & label encoding
   - Train/validation split
   - Model fine-tuning
   - Evaluation
   - Inference on `test_complaints.csv` and `submission.csv` generation

4. **Note on file paths:** If running outside Kaggle, update data file paths from `/kaggle/input/competitions/.../` to your local data directory.

---

## 🧭 Repository Structure
```
C10-team-addo/
├── README.md
├── docs/
│   ├── problem_statement.pdf
│   ├── data_card.pdf
│   ├── impact_statement_card.pdf
│   └── stakeholder_engagement.pdf
├── scripts/
│   └── complaintsense_pipeline.ipynb
└── data/
    └── [train_complaints.csv, test_complaints.csv, or download instructions]
```

---

## 👥 Appendix: Contributors & Mentors
- **Team Lead:** Esther Okoronkwo
- **Team Members:** Taiwo Uchenna, Opeyemi Precious, Kukoyi Elijah, Kingsley Ogechukwu, Peterhope Amasowoman
- **Mentor:** David Taiwo Balogun
- **Program:** AI Saturdays Lagos, Cohort 10

---

## 🔗 References
1. [ComplaintSense: Consumer Complaint Classification Challenge (Kaggle)](https://www.kaggle.com/competitions/complaint-sense-consumer-complaint-classification-challenge)
2. Hugging Face Transformers documentation — [huggingface.co/docs/transformers](https://huggingface.co/docs/transformers)
