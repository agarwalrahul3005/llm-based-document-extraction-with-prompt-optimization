import os
import json
import re
import time
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[2]

INPUT_DIR = ROOT / "data" / "ocr" / "easyocr" / "testing"
OUTPUT_DIR = ROOT / "experiments" / "predictions" / "qwen25_without_layout_aware"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BASELINE_PROMPT = """ 
You are a document understanding system. 
You are given OCR text lines from a form. 

Your job: Identify KEY-VALUE pairs. 
A key-value pair means: 
- A short label (question) 
- Followed by its corresponding value (answer) 

Return STRICT JSON: 
[ 
    {{ 
        "question": "...", 
        "answer": "..."
    }} 
]

Rules: 
- Pair nearby related text
- Questions are usually short labels 
- Do NOT hallucinate fields 
- Output ONLY JSON OCR 

TEXT: {ocr_text} 
"""


# ==== OLLAMA CALL ====
def call_ollama(prompt):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "qwen2.5:7b",
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
        match = re.search(r'\[\s*\{.*?\}\s*\]', raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except:
                pass
    return []


# ==== OCR → TEXT ====
LINE_THRESHOLD = 12
def convert_ocr_to_lines(ocr_words):
    # sort top → bottom, left → right
    words = sorted(ocr_words, key=lambda w: (w["bbox"][1], w["bbox"][0]))

    lines = []
    current_line = []
    prev_y = None

    for word in words:
        text = word["text"]
        x1, y1, x2, y2 = word["bbox"]

        if prev_y is None or abs(y1 - prev_y) < LINE_THRESHOLD:
            current_line.append((x1, text))
        else:
            # sort left → right inside line
            current_line = sorted(current_line, key=lambda x: x[0])
            lines.append(" ".join([w[1] for w in current_line]))
            current_line = [(x1, text)]

        prev_y = y1

    if current_line:
        current_line = sorted(current_line, key=lambda x: x[0])
        lines.append(" ".join([w[1] for w in current_line]))

    return lines[:50]


# ==== PROCESS FILE ====
def process_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        ocr_data = json.load(f)

    lines = convert_ocr_to_lines(ocr_data["words"])

    prompt = BASELINE_PROMPT.format(
        ocr_text="\n".join(lines)
    )

    for attempt in range(3):
        try:
            raw = call_ollama(prompt)

            print("\n--- RAW OUTPUT ---")
            print(raw)

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
    print(f"\nFound {len(files)} OCR files.\n")

    for index, file in enumerate(files):
        print("\n" + "=" * 100)
        print(f"[{index+1}/{len(files)}] {file.name}")
        print("=" * 100)

        result = process_file(file)

        with open(OUTPUT_DIR / file.name, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

    print("\n✅ Extraction complete")


if __name__ == "__main__":
    main()