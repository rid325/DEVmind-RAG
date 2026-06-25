import re

def clean_arxiv_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r'\$[^$]+\$', '', text)        
    text = re.sub(r'\\\w+\{[^}]*\}', '', text)   
    text = re.sub(r'\\\w+', '', text)             

    
    text = re.sub(r'\s+', ' ', text)

    
    text = text.encode('ascii', 'ignore').decode('ascii')

    return text.strip()


def clean_title(title: str) -> str:
    if not title:
        return ""
    text = re.sub(r'\s+', ' ', title)
    text = text.encode('ascii', 'ignore').decode('ascii')
    return text.strip()
    