import os
# ===============================
# CPU Thread Optimization
# MUST be before importing torch
# ===============================
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
torch.set_num_threads(1)
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from src.embedding_service import EmbeddingService
# ===============================
# Configuration
# ===============================
MODEL_PATH = "local_models/Tooka-SBERT-V2-Small"
# ===============================
# Global Service Holder
# ===============================
embedding_service: EmbeddingService | None = None
# ===============================
# Request Models
# ===============================
class EmbeddingRequest(BaseModel):
    texts: list[str]
    batch_size: int = 16
class EmbeddingResponse(BaseModel):
    embeddings: list[list[float]]
# ===============================
# Application Lifecycle
# ===============================
@asynccontextmanager
async def lifespan(app: FastAPI):

    global embedding_service


    print("🚀 Loading embedding model...")


    embedding_service = EmbeddingService(
        model_path=MODEL_PATH
    )


    print("🔥 Running warmup...")


    embedding_service.encode(
        [
            "warmup text"
        ]
    )


    print(
        "✅ Embedding service ready"
    )


    yield


    print(
        "🛑 Shutting down"
    )



# ===============================
# FastAPI App
# ===============================


app = FastAPI(
    title="Production Embedding Service",
    version="1.0.0",
    lifespan=lifespan
)



# ===============================
# Health Check
# ===============================


@app.get("/health")
def health():

    return {
        "status": "ok",
        "model_loaded": embedding_service is not None
    }



# ===============================
# Embedding Endpoint
# ===============================


@app.post(
    "/embed",
    response_model=EmbeddingResponse
)
def create_embedding(
    request: EmbeddingRequest
):

    if embedding_service is None:
        raise HTTPException(
            status_code=503,
            detail="Embedding model not ready"
        )


    embeddings = embedding_service.encode(
        texts=request.texts,
        batch_size=request.batch_size
    )


    return {
        "embeddings": embeddings.tolist()
    }
@app.get("/cache/status")
def cache_status():

    return {
        "cache_size":
            embedding_service.cache.size()
    }