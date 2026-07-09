# Production Embedding Service

A production-ready embedding service built with Hugging Face Transformers.

This repository documents a real-world debugging journey where a
SentenceTransformers-based embedding service failed inside Docker
while working correctly on a local machine.

The project covers:

- Debugging a silent AI service failure
- Replacing SentenceTransformers with native Transformers inference
- Building a reliable embedding pipeline
- CPU optimization techniques
- Batch processing
- Embedding caching
- Production deployment considerations

# 📖 Read the full technical article on Medium:

https://medium.com/@s.mohamad.hojabr/building-a-production-ready-embedding-service-from-sentencetransformer-to-native-transformers-8036d038ab9d

## Architecture

Initial architecture:

FastAPI
    |
    v
SentenceTransformer
    |
    v
Embedding Vector
    |
    v
ChromaDB


Final architecture:

FastAPI
    |
    v
Embedding Service
    |
    +--> AutoTokenizer
    |
    +--> AutoModel
    |
    +--> Mean Pooling
    |
    +--> Normalization
    |
    v
Embedding Vector


## Problem

The service worked locally but hung inside Docker during:

SentenceTransformer(model_path)


No exception was raised.
No memory issue occurred.
The process simply stopped progressing.


## Solution

The embedding pipeline was rewritten using:

- transformers.AutoTokenizer
- transformers.AutoModel
- custom pooling
- torch.inference_mode()


## Experiments

The repository contains reproducible experiments:

- tokenizer loading test
- model loading test
- SentenceTransformer isolation test
- performance benchmark
- memory profiling


## Model

Example model:

Tooka-SBERT-V2-Small


## License

MIT
