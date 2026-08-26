import os
import re
from . import config

def clean_text(text):
    """
    Cleans the extracted text by removing extra whitespace, fixing line breaks, etc.
    """
    # Remove multiple spaces/tabs
    text = re.sub(r"[ \t]+", " ", text)

    # Fix words broken across lines with hyphens
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    # Join wrapped lines (single newlines) into a single line
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)

    # Remove extra blank lines (more than 2 newlines)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Trim whitespace from start and end
    text = text.strip()

    return text

def clean_file(input_path, output_path):
    """
    Reads a file, cleans its content, and saves it to a new file.
    """
    print(f"Cleaning: {input_path}")
    
    if not os.path.exists(input_path):
        print(f"Input file not found: {input_path}")
        return

    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    cleaned_text = clean_text(text)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(cleaned_text)

    print(f"Cleaning Complete! Saved to: {output_path}")
    print(f"Characters: {len(cleaned_text):,}")

def main():
    if not os.path.exists(config.EXTRACTED_DIR):
        print(f"Extracted directory not found: {config.EXTRACTED_DIR}")
        return

    for filename in os.listdir(config.EXTRACTED_DIR):
        if filename.endswith(".txt"):
            input_path = os.path.join(config.EXTRACTED_DIR, filename)
            output_filename = filename.replace(".txt", "_clean.txt")
            output_path = os.path.join(config.CLEANED_DIR, output_filename)
            
            clean_file(input_path, output_path)

if __name__ == "__main__":
    main()