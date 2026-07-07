#Introduction

Building reliable embedding services is one of the most critical components in modern RAG-based AI systems. While most developers focus on model accuracy, in production environments the real challenges often emerge from system integration, performance stability, and deployment constraints rather than the model itself.

In this case study, I will walk through a real-world issue encountered while building a Persian embedding service using the Tooka-SBERT-V2-Small model inside a Dockerized FastAPI application.

The system worked flawlessly on a local machine. However, after deploying it to a development server using Docker, an unexpected issue appeared: the service would hang silently during initialization of SentenceTransformers, without throwing any errors or consuming noticeable system resources.

At first glance, the problem seemed related to memory, CPU, or model corruption. However, after a systematic debugging process, the root cause turned out to be something entirely different: the interaction between sentence-transformers, specific dependency versions, and the Docker runtime environment.

This article documents:

The original system architecture
The debugging process step-by-step
Why common assumptions (RAM, CPU, file corruption) were misleading
The actual root cause of the issue
How replacing SentenceTransformers with Hugging Face AutoModel resolved the problem
Performance improvements and production optimizations applied afterward

By the end of this article, you will see why sometimes the simplest abstraction layer can become the hardest point of failure in production systems, and why having full control over model inference can significantly improve reliability and performance.