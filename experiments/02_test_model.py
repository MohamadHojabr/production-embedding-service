from transformers import AutoModel
import time


MODEL_PATH = "../models/Tooka-SBERT-V2-Small"


print("Loading model...")

start = time.time()


model = AutoModel.from_pretrained(
    MODEL_PATH,
    local_files_only=True
)


print(
    f"Model loaded successfully in {time.time()-start:.3f}s"
)