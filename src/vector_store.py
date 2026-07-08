import faiss
import numpy as np


class VectorStore:


    def __init__(
        self,
        dimension=768
    ):

        self.dimension = dimension


        self.index = faiss.IndexFlatIP(
            dimension
        )


        self.documents = []



    def add(
        self,
        embeddings,
        documents
    ):

        vectors = np.asarray(
            embeddings,
            dtype="float32"
        )


        self.index.add(
            vectors
        )


        self.documents.extend(
            documents
        )



    def search(
        self,
        query_embedding,
        top_k=5
    ):

        query = np.asarray(
            [query_embedding],
            dtype="float32"
        )


        scores, indexes = self.index.search(
            query,
            top_k
        )


        results=[]


        for score, idx in zip(
            scores[0],
            indexes[0]
        ):

            results.append(
                {
                    "document":
                        self.documents[idx],

                    "score":
                        float(score)
                }
            )


        return results