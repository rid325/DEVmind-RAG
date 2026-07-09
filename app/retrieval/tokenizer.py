import re

def tokenize(text: str) -> list[str]:
    """
    Lowercases the text, strips punctuation (non-alphanumeric except spaces),
    and splits on whitespace.
    """
    if not text:
        return []
    
  
    text = text.lower()
    
   
    text = re.sub(r'[^\w\s]', ' ', text)
    
   
    return text.split()
