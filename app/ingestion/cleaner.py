import re
from bs4 import BeautifulSoup

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

def clean_stackoverflow_text(html: str) -> str:
    if not html:
        return ""

    soup = BeautifulSoup(html, "html.parser")

    # Wrap <pre> blocks with newlines so code stays on its own lines
    for pre in soup.find_all("pre"):
        pre.insert_before("\n")
        pre.insert_after("\n")

    # Wrap inline <code> not inside <pre> with backticks for readability
    for code in soup.find_all("code"):
        if not code.find_parent("pre"):
            code.insert_before("`")
            code.insert_after("`")

    text = soup.get_text(separator=" ")
    # Collapse excessive whitespace but preserve intentional newlines
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()
