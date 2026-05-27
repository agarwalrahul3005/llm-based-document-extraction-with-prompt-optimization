import os
import json
import re
import time
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[2]

INPUT_DIR = ROOT / "data" / "ocr_tesseract"
OUTPUT_DIR = ROOT / "data" / "predictions"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


PROMPT_TEMPLATE = """
You are a document understanding system.

You are given OCR text lines from a form.

Your job:
Identify KEY-VALUE pairs.

A key-value pair means:
- A short label (question)
- Followed by its corresponding value (answer)

Example:
"COMPOUND" → "3-Hydroxy-3-methylbutanoic acid"
"SOURCE" → "Lorillard - Organic Chemistry"

Return STRICT JSON:

[
  {{
    "question": {{"text": "...", "bbox": null}},
    "answer": {{"text": "...", "bbox": null}}
  }}
]

Rules:
- Pair nearby related text
- Questions are usually short labels
- Answers are usually longer values
- Do NOT hallucinate fields
- Output ONLY JSON

OCR TEXT:
{ocr_text}
"""


# ==== OLLAMA CALL ====
def call_ollama(prompt):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3",
            "prompt": prompt,
            "stream": False
        }
    )

    return response.json()["response"]


# ==== PARSER ====
def parse_output(raw):
    try:
        return json.loads(raw)
    except:
        match = re.search(r'\[.*\]', raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except:
                pass
    return []


# ==== OCR → TEXT ====
def convert_ocr_to_lines(ocr_words):
    lines = []
    current_line = []

    prev_y = None

    for word in ocr_words:
        text = word["text"]
        y = word["bbox"][1]

        if prev_y is None or abs(y - prev_y) < 10:
            current_line.append(text)
        else:
            lines.append(" ".join(current_line))
            current_line = [text]

        prev_y = y

    if current_line:
        lines.append(" ".join(current_line))

    return lines[:50]  # limit


# ==== PROCESS FILE ====
def process_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        ocr_data = json.load(f)

    lines = convert_ocr_to_lines(ocr_data["words"])

    print("\n--- SAMPLE OCR LINES ---")
    for l in lines[:5]:
        print(l)

    prompt = PROMPT_TEMPLATE.format(
        ocr_text="\n".join(lines)
    )

    for attempt in range(3):
        try:
            raw = call_ollama(prompt)
            parsed = parse_output(raw)

            if parsed:
                return parsed

        except Exception as e:
            print(f"Retry {attempt+1} failed:", e)
            time.sleep(2)

    return []


# ==== MAIN ====
def main():
    files = list(INPUT_DIR.glob("*.json"))

    print(f"Processing {len(files)} files...")

    for file in files:
        print(f"\n→ {file.name}")

        result = process_file(file)

        with open(OUTPUT_DIR / file.name, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

    print("\n✅ Extraction complete")


if __name__ == "__main__":
    main()