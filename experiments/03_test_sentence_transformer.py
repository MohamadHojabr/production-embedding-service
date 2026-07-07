from sentence_transformers import SentenceTransformer
import time


MODEL_PATH = "../models/Tooka-SBERT-V2-Small"


print("Loading SentenceTransformer...")


start = time.time()


model = SentenceTransformer(
    MODEL_PATH,
    device="cpu"
)


print(
    f"Loaded in {time.time()-start:.3f}s"
)