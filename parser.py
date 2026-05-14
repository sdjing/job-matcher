import re
import sys
from pdfminer.high_level import extract_text


def load_resume(filepath):
    if filepath.endswith(".pdf"):
        raw = extract_text(filepath)
    elif filepath.endswith(".txt"):
        with open(filepath, "r", encoding="utf-8") as f:
            raw = f.read()
    else:
        print("Unsupported file type. Use a .pdf or .txt file.")
        sys.exit(1)

    return clean_text(raw)


def clean_text(text):
    text = re.sub(r"\s+", " ", text)        # collapse whitespace
    text = re.sub(r"[^\x00-\x7F]+", " ", text)  # strip non-ASCII (PDF artifacts)
    return text.strip().lower()


def preview(text, chars=500):
    print("\n--- Resume Preview ---")
    print(text[:chars])
    print("...")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python parser.py <path_to_resume.pdf>")
        sys.exit(1)

    resume_text = load_resume(sys.argv[1])
    preview(resume_text)
    print(f"\nTotal characters extracted: {len(resume_text)}")