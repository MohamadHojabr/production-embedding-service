import torch
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModel
)

from .pooling import mean_pooling
from .cache import EmbeddingCache
from .utils import generate_text_hash


class EmbeddingService:


    _instance = None

    def __new__(cls, *args, **kwargs):

        if cls._instance is None:
            cls._instance = super().__new__(cls)

        return cls._instance

    def __init__(
        self,
        model_path: str
    ):

        self.model_path = model_path

        self.device = "cpu"


        self.cache = EmbeddingCache(
            max_size=10000
        )


        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True
        )


        self.model = AutoModel.from_pretrained(
            model_path,
            local_files_only=True
        )


        self.model.eval()
    def encode(
        self,
        texts,
        batch_size=16
    ):

        embeddings = []

        missing_texts = []

        missing_indexes = []


        # =====================
        # Check Cache
        # =====================

        for idx, text in enumerate(texts):

            key = generate_text_hash(text)

            cached = self.cache.get(key)


            if cached is not None:

                embeddings.append(
                    cached
                )

            else:

                embeddings.append(
                    None
                )

                missing_texts.append(
                    text
                )

                missing_indexes.append(
                    idx
                )


        # =====================
        # Encode Missing Items
        # =====================

        if missing_texts:


            generated = self._encode_batch(
                missing_texts,
                batch_size
            )


            for idx, emb, text in zip(
                missing_indexes,
                generated,
                missing_texts
            ):

                key = generate_text_hash(
                    text
                )


                self.cache.set(
                    key,
                    emb
                )


                embeddings[idx] = emb



        return np.vstack(
            embeddings
        )
    
    def _encode_batch(
    self,
    texts,
    batch_size
    ):

        result = []


        for i in range(
            0,
            len(texts),
            batch_size
        ):

            batch = texts[
                i:i+batch_size
            ]


            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )


            inputs = {
                k:v.to(self.device)
                for k,v in inputs.items()
            }


            with torch.inference_mode():

                outputs = self.model(
                    **inputs
                )


                embeddings = mean_pooling(
                    outputs,
                    inputs["attention_mask"]
                )


                embeddings = torch.nn.functional.normalize(
                    embeddings,
                    p=2,
                    dim=1
                )


            result.extend(
                embeddings.cpu().numpy()
            )


        return np.array(
            result,
            dtype=np.float32
        )