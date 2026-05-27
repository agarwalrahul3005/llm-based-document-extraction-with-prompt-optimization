import os
import json
import re
import time
from pathlib import Path

# ==== CONFIG ====
USE_OPENAI = True   # set False if using Ollama

INPUT_DIR = Path("data/ocr")
OUTPUT_DIR = Path("data/predictions")
OUTPUT_DIR.mkdir(exist_ok=True)

# ==== PROMPT ====
PROMPT_TEMPLATE = """
You are an intelligent document understanding system.

Extract ALL key-value pairs from the document.

Return STRICT JSON list:

[
  {
    "question": {"text": "...", "bbox": [x1,y1,x2,y2]},
    "answer": {"text": "...", "bbox": [x1,y1,x2,y2]}
  }
]

Rules:
- Use only given data
- Do not hallucinate
- Preserve text exactly
- bbox can be null if unsure
- Output ONLY JSON

OCR DATA:
{ocr_json}
"""

# ==== LLM CALL ====
def call_openai(prompt):
    from openai import OpenAI
    client = OpenAI()

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )

    return response.choices[0].message.content


def call_ollama(prompt):
    import requests

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


# ==== MAIN LOGIC ====
def process_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        ocr_data = json.load(f)

    prompt = PROMPT_TEMPLATE.format(
        ocr_json=json.dumps(ocr_data["words"][:200], indent=2)  # limit size
    )

    # Retry logic (LLMs fail randomly)
    for attempt in range(3):
        try:
            if USE_OPENAI:
                raw_output = call_openai(prompt)
            else:
                raw_output = call_ollama(prompt)

            parsed = parse_output(raw_output)

            if parsed:
                return parsed

        except Exception as e:
            print(f"Retry {attempt+1} failed:", e)
            time.sleep(2)

    return []


def main():
    files = list(INPUT_DIR.glob("*.json"))

    print(f"Processing {len(files)} files...")

    for file in files:
        print(f"→ {file.name}")

        result = process_file(file)

        output_file = OUTPUT_DIR / file.name

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

    print("✅ Extraction complete")


if __name__ == "__main__":
    main()