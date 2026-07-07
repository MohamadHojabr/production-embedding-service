from transformers import AutoTokenizer
import time


MODEL_PATH = "../models/Tooka-SBERT-V2-Small"


print("Loading tokenizer...")

start = time.time()

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True
)


print(
    f"Tokenizer loaded successfully in {time.time()-start:.3f}s"
)