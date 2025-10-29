# clinical-nlp-ner

Clinical named entity recognition pipeline inspired by the "Natural Language Processing for Smart Healthcare" review. The goal is to demonstrate a full-stack workflow for extracting biomedical entities (diseases, chemicals) from text using BioBERT and the BC5CDR dataset.

## Project Layout

```
clinical-nlp-ner/
?? README.md
?? pyproject.toml
?? src/
?  ?? data.py
?  ?? model.py
?  ?? train.py
?  ?? eval.py
?  ?? infer.py
?  ?? app.py
?? configs/
?  ?? ner_biobert.yaml
?? tests/
?  ?? test_data.py
?  ?? test_model.py
?  ?? test_infer.py
?? scripts/
   ?? download_bc5cdr.sh
   ?? run_train.sh
   ?? run_eval.sh
```

## Quickstart

1. **Create an environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```
2. **Install dependencies**
   ```bash
   pip install -e .
   ```
3. **Download BC5CDR data**
   ```bash
   bash scripts/download_bc5cdr.sh
   ```
4. **Train BioBERT**
   ```bash
   bash scripts/run_train.sh configs/ner_biobert.yaml
   ```
5. **Evaluate the model**
   ```bash
   bash scripts/run_eval.sh configs/ner_biobert.yaml
   ```
6. **Try interactive demo**
   ```bash
   python -m src.app
   ```

## Status

All modules currently contain minimal scaffolding and raise `NotImplementedError`. Fill in each component following the clinical NLP pipeline (preprocessing, feature extraction, modeling, application) described in the target paper.
