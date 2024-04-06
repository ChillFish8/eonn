import json

import numpy as np
from sentence_transformers import SentenceTransformer

movies = json.load(open("./movies.json", encoding="utf-8"))

encoder = SentenceTransformer('intfloat/multilingual-e5-large')

embeddings = encoder.encode(movies, show_progress_bar=True, normalize_embeddings=True)

with open("./movies-encoded.json", "w+", encoding="utf-8") as file:
    json.dump(embeddings.astype(np.float32).tolist(), file)
