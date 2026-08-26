import re

def detect_chapters(text):
    """
    Detects chapters in the text using common patterns.
    Currently returns the text as a single 'chapter' if no patterns are found.
    """

    # Simple regex for common chapter headings
    chapter_pattern = re.compile(r'^(?:CHAPTER|Chapter|Part)\s+(?:[IVXLCDM]+|\d+)', re.MULTILINE)
    
    matches = list(chapter_pattern.finditer(text))
    
    if not matches:
        return [{"title": "Full Text", "content": text}]
    
    chapters = []
    for i in range(len(matches)):
        start = matches[i].start()
        end = matches[i+1].start() if i + 1 < len(matches) else len(text)
        
        title = text[start:text.find('\n', start)].strip()
        content = text[start:end].strip()
        
        chapters.append({
            "title": title,
            "content": content
        })
        
    return chapters

def main():
    print("Chapter detection module loaded.")

if __name__ == "__main__":
    main()
