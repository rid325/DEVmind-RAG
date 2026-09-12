"""Questions frozen before candidate retrieval for the ranking benchmark.

Do not edit these queries after pooling candidates. Create a new benchmark
version instead, otherwise the relevance judgments no longer describe the
same evaluation task.
"""

HELDOUT_VERSION = "devmind-heldout-20-v1"

HELDOUT_QUESTIONS = [
    {"id": "h01", "query": "When should I choose an HNSW index instead of IVFFlat in pgvector?", "domain": "github", "difficulty": "medium"},
    {"id": "h02", "query": "Why can a filtered approximate nearest-neighbor query return fewer rows than requested?", "domain": "github", "difficulty": "hard"},
    {"id": "h03", "query": "How should I tune IVFFlat probes and lists for recall and speed?", "domain": "github", "difficulty": "medium"},
    {"id": "h04", "query": "Can pgvector combine vector similarity with ordinary PostgreSQL filters?", "domain": "github", "difficulty": "easy"},
    {"id": "h05", "query": "How can I configure retries and timeouts in the OpenAI Python client?", "domain": "github", "difficulty": "easy"},
    {"id": "h06", "query": "What is the difference between streaming and non-streaming OpenAI responses?", "domain": "github", "difficulty": "medium"},
    {"id": "h07", "query": "Why does a PyTorch tensor sometimes need contiguous() before view()?", "domain": "stackoverflow", "difficulty": "medium"},
    {"id": "h08", "query": "What is the correct order for zero_grad, backward, and optimizer step in PyTorch?", "domain": "stackoverflow", "difficulty": "easy"},
    {"id": "h09", "query": "Why might torch.cuda.empty_cache() not solve a CUDA out-of-memory error?", "domain": "stackoverflow", "difficulty": "hard"},
    {"id": "h10", "query": "When should a Python method use classmethod rather than staticmethod?", "domain": "stackoverflow", "difficulty": "medium"},
    {"id": "h11", "query": "How does pack_padded_sequence help process variable-length batches?", "domain": "stackoverflow", "difficulty": "hard"},
    {"id": "h12", "query": "Why should model.eval() be called during neural network inference?", "domain": "stackoverflow", "difficulty": "easy"},
    {"id": "h13", "query": "How does retrieval-augmented generation reduce hallucinations?", "domain": "arxiv", "difficulty": "easy"},
    {"id": "h14", "query": "What makes evaluating an industrial RAG system difficult?", "domain": "arxiv", "difficulty": "medium"},
    {"id": "h15", "query": "How does late chunking differ from contextual retrieval?", "domain": "arxiv", "difficulty": "hard"},
    {"id": "h16", "query": "What security risks arise when retrieved documents contain malicious instructions?", "domain": "arxiv", "difficulty": "medium"},
    {"id": "h17", "query": "Why does standard attention become expensive for long input sequences?", "domain": "arxiv", "difficulty": "easy"},
    {"id": "h18", "query": "How can iterative retrieval recover evidence missed by the first search?", "domain": "arxiv", "difficulty": "hard"},
    {"id": "h19", "query": "What are the tradeoffs between dense and sparse retrieval for technical questions?", "domain": "cross-domain", "difficulty": "hard"},
    {"id": "h20", "query": "How should a technical assistant preserve source provenance from ingestion through generation?", "domain": "cross-domain", "difficulty": "hard"},
]
