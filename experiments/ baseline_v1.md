# Baseline V1

Date
2026-07-22

----------------------------

Model

Llama 3 8B
(Ollama)

----------------------------

Framework

DSPy

----------------------------

OCR

Tesseract OCR

----------------------------

Document Representation

• OCR words

• Lines

• Line bounding boxes

• Word bounding boxes

----------------------------

Prompt

Baseline Prompt V1

- Extract key-value pairs
- Use OCR text
- Use layout
- Return JSON only
- No hallucination

----------------------------

Parser

Regex JSON parser

----------------------------

Evaluator

Fuzzy Matching

Threshold = 0.55

Metrics

Precision

Recall

F1

----------------------------

Results

Precision : 0.xx

Recall : 0.xx

F1 : 0.xx