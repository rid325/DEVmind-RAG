import re

def tokenize(text: str) -> list[str]:
    """
    Lowercases the text, strips punctuation (non-alphanumeric except spaces),
    and splits on whitespace.
    """
    if not text:
        return []
    
    # Lowercase the text
    text = text.lower()
    
    # Replace anything that isn't a word character (\w) or whitespace (\s) with a space
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Split on whitespace
    return text.split()
