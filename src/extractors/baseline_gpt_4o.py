import json
import re
import time
from pathlib import Path

from openai import OpenAI

client = OpenAI()

# ================= PATH SETUP =================
ROOT = Path(__file__).resolve().parents[2]

INPUT_DIR = ROOT / "data" / "ocr_tesseract"
OUTPUT_DIR = ROOT / "data" / "predictions_openapi_training"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ================= OCR → LINE GROUPING =================
def group_words_into_lines(words, y_threshold=10):
    """
    Groups words into lines based on vertical proximity (y-axis)
    """
    words = sorted(words, key=lambda w: (w["bbox"][1], w["bbox"][0]))

    lines = []
    current_line = []
    current_y = None

    for w in words:
        y = w["bbox"][1]

        if current_y is None:
            current_line.append(w)
            current_y = y
        elif abs(y - current_y) <= y_threshold:
            current_line.append(w)
        else:
            lines.append(current_line)
            current_line = [w]
            current_y = y

    if current_line:
        lines.append(current_line)

    # Convert to text lines
    line_texts = []
    for line in lines:
        line = sorted(line, key=lambda w: w["bbox"][0])
        text = " ".join([w["text"] for w in line])
        line_texts.append(text)

    return line_texts


# ================= PROMPT =================
def build_prompt(lines):
    return f"""
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
- Do NOT return empty unless nothing found
- Output ONLY JSON

OCR TEXT:
{json.dumps(lines, indent=2)}
"""


# ================= LLM CALL =================
def call_llm(prompt):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content


# ================= PARSER =================
def parse_output(raw):
    try:
        return json.loads(raw)
    except:
        match = re.search(r'\[\s*{.*}\s*\]', raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except:
                pass
    return []


# ================= MAIN PROCESS =================
def process_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        ocr_data = json.load(f)

    words = ocr_data.get("words", [])

    # 🔥 Convert to lines (IMPORTANT FIX)
    lines = group_words_into_lines(words)

    # Limit size (LLM token control)
    lines = lines[:50]

    prompt = build_prompt(lines)

    # DEBUG (don’t skip this)
    print("\n--- SAMPLE LINES ---")
    for l in lines[:5]:
        print(l)

    for attempt in range(3):
        try:
            raw = call_llm(prompt)

            print("\n--- RAW LLM OUTPUT ---")
            print(raw[:500])

            parsed = parse_output(raw)

            if parsed:
                return parsed

        except Exception as e:
            print(f"Retry {attempt+1} failed:", e)
            time.sleep(2)

    return []


# ================= MAIN =================
def main():
    files = list(INPUT_DIR.glob("*.json"))

    print(f"Processing {len(files)} files...")

    for file in files[:3]:  # ⚠️ start small
        print(f"\n→ {file.name}")

        result = process_file(file)

        with open(OUTPUT_DIR / file.name, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

    print("\n✅ Extraction complete")


if __name__ == "__main__":
    main()